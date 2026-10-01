#include "hex-common.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#include "dma-queue.h"
#include "htp-vtcm.h"
#include "hvx-utils.h"
#include "hex-fastdiv.h"
#include <string.h>

struct htp_concat_context {
    struct htp_ops_context * octx;
    uint32_t dim;
    uint32_t nrows_per_thread;
    uint32_t row_start;
    uint32_t nrows;
    uint32_t elem_start;
    uint32_t nelems;
    uint32_t nplanes;
    struct fastdiv_values div_ne0;
    struct fastdiv_values div_ne1;
    struct fastdiv_values div_ne2;
};

static void concat_2d_f32_transposed(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t src0_ne0 = src0->ne[0];
    const uint32_t src1_ne0 = src1->ne[0];

    const uint32_t row_end = cctx->row_start + cctx->nrows;
    const uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    const uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    if (start_i >= end_i) return;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * spad1_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;

    const uint32_t block_i = 32;
    const uint32_t spad1_stride = block_i * sizeof(float);

    int32_t offsets[32] __attribute__((aligned(128)));
    for(int k=0; k<32; k++) {
        offsets[k] = k * spad1_stride;
    }
    HVX_Vector vv = *(HVX_Vector*)offsets;
    const uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 32);
    const uint32_t spad0_row_bytes = hex_round_up((src0_ne0 + src1_ne0_padded) * sizeof(float), VLEN);
    uint32_t mu = src1_ne0_padded * spad1_stride;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    for (uint32_t p = 0; p < cctx->nplanes; p++) {
        const uint32_t i3 = p / dst->ne[2];
        const uint32_t i2 = p - i3 * dst->ne[2];
        const dma_addr_t src0_plane = src0->data + i2 * src0->nb[2] + i3 * src0->nb[3];
        const dma_addr_t src1_plane = src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
        const dma_addr_t dst_plane  = dst->data  + i2 * dst->nb[2]  + i3 * dst->nb[3];

        for (uint32_t i = start_i; i < end_i; i += block_i) {
            uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;

            uint32_t src1_width_bytes = current_block_i * sizeof(float);
            const dma_addr_t src1_addr = src1_plane + i * src1->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], src1_width_bytes, src1_ne0);

            uint32_t src0_row_bytes = src0_ne0 * sizeof(float);
            const dma_addr_t src0_addr = src0_plane + i * src0->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad0_base, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, current_block_i);

            dma_queue_pop(dma_q); // src1

            HVX_Vector * vtcm_tmp = (HVX_Vector *)(spad1_base + src1_ne0_padded * spad1_stride);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            for (uint32_t j = 0; j < src1_ne0_padded; j += 32) {
                #pragma unroll(4)
                for (uint32_t ii = 0; ii < current_block_i; ii++) {
                    size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(float));
                    Q6_vgather_ARMVw(&vtcm_tmp[ii], rt, mu, vv);
                    uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(float);
                    hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

            dma_queue_pop(dma_q); // src0

            const dma_addr_t dst_addr = dst_plane + i * dst->nb[1];
            dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(float), current_block_i);

            dma_queue_pop(dma_q);
        }
    }
}

static void concat_2d_f16_transposed(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t src0_ne0 = src0->ne[0];
    const uint32_t src1_ne0 = src1->ne[0];

    const uint32_t row_end = cctx->row_start + cctx->nrows;
    const uint32_t start_i = cctx->row_start + ith * cctx->nrows_per_thread;
    const uint32_t end_i   = (start_i + cctx->nrows_per_thread < row_end) ? (start_i + cctx->nrows_per_thread) : row_end;
    if (start_i >= end_i) return;

    dma_queue * dma_q = octx->ctx->dma[ith];

    uint8_t * spad0_base = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;
    uint8_t * spad1_base = octx->src1_spad.data + ith * octx->src1_spad.size_per_thread;

    const uint32_t block_i = 64;
    const uint32_t spad1_stride = block_i * sizeof(__fp16);

    int16_t offsets[64] __attribute__((aligned(128)));
    for(int k=0; k<64; k++) {
        offsets[k] = k * spad1_stride;
    }
    HVX_Vector vv = *(HVX_Vector*)offsets;
    const uint32_t src1_ne0_padded = hex_round_up(src1_ne0, 64);
    const uint32_t spad0_row_bytes = hex_round_up((src0_ne0 + src1_ne0_padded) * sizeof(__fp16), VLEN);
    uint32_t mu = src1_ne0_padded * spad1_stride;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    for (uint32_t p = 0; p < cctx->nplanes; p++) {
        const uint32_t i3 = p / dst->ne[2];
        const uint32_t i2 = p - i3 * dst->ne[2];
        const dma_addr_t src0_plane = src0->data + i2 * src0->nb[2] + i3 * src0->nb[3];
        const dma_addr_t src1_plane = src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
        const dma_addr_t dst_plane  = dst->data  + i2 * dst->nb[2]  + i3 * dst->nb[3];

        for (uint32_t i = start_i; i < end_i; i += block_i) {
            uint32_t current_block_i = (end_i - i < block_i) ? (end_i - i) : block_i;

            uint32_t src1_width_bytes = current_block_i * sizeof(__fp16);
            const dma_addr_t src1_addr = src1_plane + i * src1->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad1_base, src1_addr), spad1_stride, src1->nb[0], src1_width_bytes, src1_ne0);

            uint32_t src0_row_bytes = src0_ne0 * sizeof(__fp16);
            const dma_addr_t src0_addr = src0_plane + i * src0->nb[1];
            dma_queue_push(dma_q, dma_make_data(spad0_base, src0_addr), spad0_row_bytes, src0->nb[1], src0_row_bytes, current_block_i);

            dma_queue_pop(dma_q); // src1

            HVX_Vector * vtcm_tmp = (HVX_Vector *)(spad1_base + src1_ne0_padded * spad1_stride);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            for (uint32_t j = 0; j < src1_ne0_padded; j += 64) {
                #pragma unroll(4)
                for (uint32_t ii = 0; ii < current_block_i; ii++) {
                    size_t rt = (size_t)(spad1_base + j * spad1_stride + ii * sizeof(__fp16));
                    Q6_vgather_ARMVh(&vtcm_tmp[ii], rt, mu, vv);
                    uint8_t * dst_ptr = spad0_base + ii * spad0_row_bytes + (src0_ne0 + j) * sizeof(__fp16);
                    hvx_vmemu(dst_ptr) = vtcm_tmp[ii];
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);

            dma_queue_pop(dma_q); // src0

            const dma_addr_t dst_addr = dst_plane + i * dst->nb[1];
            dma_queue_push(dma_q, dma_make_data(dst_addr, spad0_base), dst->nb[1], spad0_row_bytes, (src0_ne0 + src1_ne0) * sizeof(__fp16), current_block_i);

            dma_queue_pop(dma_q);
        }
    }
}

static void concat_generic(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const int dim = cctx->dim;
    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;

    const uint32_t ne[4] = {dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]};

    // Per-device element range aligned to prevent false sharing
    const uint32_t elem_start = cctx->elem_start;
    const uint32_t nelems     = cctx->nelems;
    const uint32_t chunk_size = (nelems + nth - 1) / nth;

    const uint32_t start_idx = MIN(elem_start + ith * chunk_size, elem_start + nelems);
    const uint32_t end_idx   = MIN(start_idx + chunk_size, elem_start + nelems);

    // Naive scalar element-wise copy
    for (uint32_t idx = start_idx; idx < end_idx; idx++) {
        uint32_t idx_div_ne0 = fastdiv(idx, &cctx->div_ne0);
        uint32_t i0 = idx - idx_div_ne0 * ne[0];

        uint32_t idx_div_ne01 = fastdiv(idx_div_ne0, &cctx->div_ne1);
        uint32_t i1 = idx_div_ne0 - idx_div_ne01 * ne[1];

        uint32_t idx_div_ne012 = fastdiv(idx_div_ne01, &cctx->div_ne2);
        uint32_t i2 = idx_div_ne01 - idx_div_ne012 * ne[2];
        uint32_t i3 = idx_div_ne012;

        uint8_t * dst_ptr = (uint8_t *)dst->data + i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1] + i0 * dst->nb[0];

        uint32_t idx_dim = 0;
        if (dim == 0) idx_dim = i0;
        else if (dim == 1) idx_dim = i1;
        else if (dim == 2) idx_dim = i2;
        else if (dim == 3) idx_dim = i3;

        const struct htp_tensor * src = (idx_dim < src0->ne[dim]) ? src0 : src1;

        uint32_t s0 = i0;
        uint32_t s1 = i1;
        uint32_t s2 = i2;
        uint32_t s3 = i3;

        if (dim == 0 && src == src1) s0 -= src0->ne[0];
        if (dim == 1 && src == src1) s1 -= src0->ne[1];
        if (dim == 2 && src == src1) s2 -= src0->ne[2];
        if (dim == 3 && src == src1) s3 -= src0->ne[3];

        uint8_t * src_ptr = (uint8_t *)src->data + s3 * src->nb[3] + s2 * src->nb[2] + s1 * src->nb[1] + s0 * src->nb[0];

        if (type_size == 4) {
            *(float*)dst_ptr = *(float*)src_ptr;
        } else {
            *(__fp16*)dst_ptr = *(__fp16*)src_ptr;
        }
    }
}

static bool concat_dim1_contiguous_dma(struct htp_ops_context * octx, int dim, uint32_t type_size) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (dim != 1 || octx->ctx->mdev.count > 1 ||
        (dst->type != HTP_TYPE_F32 && dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_I32) ||
        src0->type != dst->type || src1->type != dst->type ||
        src0->ne[0] != dst->ne[0] || src1->ne[0] != dst->ne[0] ||
        src0->ne[2] != dst->ne[2] || src1->ne[2] != dst->ne[2] ||
        src0->ne[3] != dst->ne[3] || src1->ne[3] != dst->ne[3] ||
        dst->ne[1] != src0->ne[1] + src1->ne[1] ||
        !htp_tensor_is_contiguous(src0, type_size) ||
        !htp_tensor_is_contiguous(src1, type_size) ||
        !htp_tensor_is_contiguous(dst, type_size)) {
        return false;
    }

    const uint32_t src0_row_size = src0->ne[0] * type_size;
    const uint32_t src1_row_size = src1->ne[0] * type_size;

    // v75+ dma_queue_push() writes a 2D descriptor directly and does not split overflow.
#if __HVX_ARCH__ >= 75
    if (src0_row_size > 0xffffffu || src1_row_size > 0xffffffu ||
        src0->nb[1] > 0xffffffu || src1->nb[1] > 0xffffffu || dst->nb[1] > 0xffffffu ||
        src0->ne[1] > UINT16_MAX || src1->ne[1] > UINT16_MAX) {
        return false;
    }
#endif

    dma_queue * q = octx->ctx->dma[0];

    for (uint32_t i3 = 0; i3 < dst->ne[3]; ++i3) {
        for (uint32_t i2 = 0; i2 < dst->ne[2]; ++i2) {
            dma_addr_t dst_addr  = dst->data  + i3 * dst->nb[3]  + i2 * dst->nb[2];
            dma_addr_t src0_addr = src0->data + i3 * src0->nb[3] + i2 * src0->nb[2];
            dma_addr_t src1_addr = src1->data + i3 * src1->nb[3] + i2 * src1->nb[2];

            if (!dma_queue_push(q, dma_make_data(dst_addr, src0_addr), dst->nb[1], src0->nb[1], src0_row_size, src0->ne[1])) {
                dma_queue_flush(q);
                dma_queue_push(q, dma_make_data(dst_addr, src0_addr), dst->nb[1], src0->nb[1], src0_row_size, src0->ne[1]);
            }

            dst_addr += src0->ne[1] * dst->nb[1];
            if (!dma_queue_push(q, dma_make_data(dst_addr, src1_addr), dst->nb[1], src1->nb[1], src1_row_size, src1->ne[1])) {
                dma_queue_flush(q);
                dma_queue_push(q, dma_make_data(dst_addr, src1_addr), dst->nb[1], src1->nb[1], src1_row_size, src1->ne[1]);
            }
        }
    }

    dma_queue_flush(q);
    return true;
}

// dim 0 concat with short output rows (W = ne00 + ne10 divides 32) and src1 contiguous along rows
// (e.g. the Gated DeltaNet conv state [3, C] + transposed token [1, C]): one DMA per input, vgather, one DMA out
static void concat_dim0_gather_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context * octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t ne00  = src0->ne[0];
    const uint32_t ne10  = src1->ne[0];
    const uint32_t W     = ne00 + ne10;
    const uint32_t nrows = dst->ne[1];
    const uint32_t r0    = ith * cctx->nrows_per_thread;
    if (r0 >= nrows) {
        return;
    }
    const uint32_t nr = MIN(cctx->nrows_per_thread, nrows - r0);

    dma_queue * dma_q = octx->ctx->dma[ith];
    uint8_t * spad = octx->src0_spad.data + ith * octx->src0_spad.size_per_thread;

    const uint32_t a_bytes = hex_round_up(cctx->nrows_per_thread * ne00 * sizeof(float), VLEN);
    const uint32_t b_bytes = hex_round_up(cctx->nrows_per_thread * ne10 * sizeof(float), VLEN);
    uint8_t * va = spad;
    uint8_t * vb = spad + a_bytes;
    HVX_Vector * vo = (HVX_Vector *) (vb + b_bytes);

    // word offsets (from va) for output vector 0, and the increment per output vector
    const uint32_t rpv = 32 / W;
    int32_t off0[32] __attribute__((aligned(128)));
    int32_t step[32] __attribute__((aligned(128)));
    for (uint32_t l = 0; l < 32; l++) {
        const uint32_t r = l / W, e = l % W;
        if (e < ne00) {
            off0[l] = (r * ne00 + e) * sizeof(float);
            step[l] = rpv * ne00 * sizeof(float);
        } else {
            off0[l] = a_bytes + ((e - ne00) * nr + r) * sizeof(float);
            step[l] = rpv * sizeof(float);
        }
    }
    const HVX_Vector v_step = *(const HVX_Vector *) step;
    const uint32_t mu = a_bytes + b_bytes - 1;
    const uint32_t n_vec = (nr * W + 31) / 32;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    for (uint32_t p = 0; p < cctx->nplanes; p++) {
        const uint32_t i3 = p / dst->ne[2];
        const uint32_t i2 = p - i3 * dst->ne[2];
        const dma_addr_t src0_plane = src0->data + i2 * src0->nb[2] + i3 * src0->nb[3];
        const dma_addr_t src1_plane = src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
        const dma_addr_t dst_plane  = dst->data  + i2 * dst->nb[2]  + i3 * dst->nb[3];

        dma_queue_push_single_1d(dma_q, dma_make_data(va, src0_plane + r0 * src0->nb[1]), nr * ne00 * sizeof(float));
        for (uint32_t j = 0; j < ne10; j++) {
            dma_queue_push_single_1d(dma_q, dma_make_data(vb + j * nr * sizeof(float), src1_plane + j * src1->nb[0] + r0 * src1->nb[1]),
                                     nr * sizeof(float));
        }
        for (uint32_t j = 0; j < 1 + ne10; j++) {
            dma_queue_pop(dma_q);
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) p);
        HVX_Vector v_off = *(const HVX_Vector *) off0;
        for (uint32_t k = 0; k < n_vec; k++) {
            Q6_vgather_ARMVw(&vo[k], (size_t) va, mu, v_off);
            v_off = Q6_Vw_vadd_VwVw(v_off, v_step);
        }
        // a vgather writes its destination asynchronously: a vector load of the destination waits for it, the DMA
        // below does not, so it could read vectors whose gathers are still in flight
        for (uint32_t k = 0; k < n_vec; k++) {
            (void) *(volatile HVX_Vector *) &vo[k];
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) p);

        dma_queue_push_single_1d(dma_q, dma_make_data(dst_plane + r0 * dst->nb[1], vo), nr * W * sizeof(float));
        dma_queue_pop(dma_q);
    }
}

static inline void concat_push(dma_queue * q, dma_data d, size_t dst_stride, size_t src_stride, size_t row, size_t n) {
    if (!dma_queue_push(q, d, dst_stride, src_stride, row, n)) {
        dma_queue_flush(q);
        dma_queue_push(q, d, dst_stride, src_stride, row, n);
    }
}

// "chanfast" Gated DeltaNet conv input (the host rewrites the conv_input tensor as channel-fastest, see
// ggml-hexagon.cpp): dst row t = ne1 floats (nb[1] == 4, nb[0] == ne1*4). Rows [0, ne00) = src0 (the conv state
// [ne00, C], time fastest) transposed in VTCM; rows [ne00, ne0) = src1 (the tokens: a transposed view of a
// channel-fastest tensor, nb[1] == 4) copied DDR -> DDR by one 2D DMA per thread and plane. No per-element transpose.
static void concat_chanfast_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_concat_context * cctx = (struct htp_concat_context *) data;
    struct htp_ops_context *    octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    const uint32_t C  = dst->ne[1];
    const uint32_t c0 = ith * cctx->nrows_per_thread;
    const uint32_t c1 = MIN(c0 + cctx->nrows_per_thread, C);
    if (c0 >= c1) {
        return;
    }
    const uint32_t cw  = c1 - c0;
    const uint32_t cws = hex_round_up(cw, 32);
    const uint32_t ns0 = src0->ne[0];
    const uint32_t nt  = src1->ne[0];

    dma_queue * q   = octx->ctx->dma[ith];
    float *     raw = (float *) (octx->src0_spad.data + ith * octx->src0_spad.size_per_thread);
    float *     T   = raw + hex_round_up(cw * ns0, 32);

    for (uint32_t i3 = 0; i3 < dst->ne[3]; ++i3) {
        for (uint32_t i2 = 0; i2 < dst->ne[2]; ++i2) {
            const dma_addr_t d_plane  = dst->data  + i3 * dst->nb[3]  + i2 * dst->nb[2]  + c0 * sizeof(float);
            const dma_addr_t s0_slice = src0->data + i3 * src0->nb[3] + i2 * src0->nb[2] + c0 * src0->nb[1];
            const dma_addr_t s1_slice = src1->data + i3 * src1->nb[3] + i2 * src1->nb[2] + c0 * sizeof(float);
            const size_t     st_bytes = (size_t) cw * ns0 * sizeof(float);

            // queue: [state in, tokens] -> pop state in, transpose, [tokens, state out] -> pop both
            concat_push(q, dma_make_data((uint8_t *) raw, s0_slice), st_bytes, st_bytes, st_bytes, 1);
            concat_push(q, dma_make_data(d_plane + (size_t) ns0 * dst->nb[0], s1_slice), dst->nb[0], src1->nb[0],
                        (size_t) cw * sizeof(float), nt);
            dma_queue_pop(q);

            for (uint32_t c = 0; c < cw; ++c) {
                for (uint32_t j = 0; j < ns0; ++j) {
                    T[(size_t) j * cws + c] = raw[(size_t) c * ns0 + j];
                }
            }

            concat_push(q, dma_make_data(d_plane, (uint8_t *) T), dst->nb[0], (size_t) cws * sizeof(float),
                        (size_t) cw * sizeof(float), ns0);
            dma_queue_pop(q);
            dma_queue_pop(q);
        }
    }
}

static bool concat_is_chanfast(struct htp_ops_context * octx, int dim) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    return dim == 0 && octx->ctx->mdev.count <= 1 &&
           dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F32 && src1->type == HTP_TYPE_F32 &&
           dst->nb[1] == 4 && dst->nb[0] == (size_t) dst->ne[1] * 4 &&
           src0->nb[0] == 4 && src0->nb[1] == (size_t) src0->ne[0] * 4 &&
           src1->nb[1] == 4 && src1->nb[0] >= (size_t) src1->ne[1] * 4 &&
           src0->ne[1] == dst->ne[1] && src1->ne[1] == dst->ne[1] && dst->ne[0] == src0->ne[0] + src1->ne[0] &&
           !htp_tensor_is_extended(src0) && !htp_tensor_is_extended(src1) && !htp_tensor_is_extended(dst);
}

int op_concat(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    int dim = octx->op_params[0];

    const uint32_t type_size = (dst->type == HTP_TYPE_F32 || dst->type == HTP_TYPE_I32) ? 4 : 2;
    bool is_src1_transposed  = (src1->nb[0] > src1->nb[1]);
    bool is_src0_transposed  = (src0->nb[0] > src0->nb[1]);

    if (concat_is_chanfast(octx, dim)) {
        struct htp_concat_context cctx;
        const uint32_t n_threads = octx->n_threads;
        cctx.octx             = octx;
        cctx.dim              = dim;
        cctx.nrows_per_thread = hex_round_up((dst->ne[1] + n_threads - 1) / n_threads, 32);
        const uint32_t cws    = cctx.nrows_per_thread;
        octx->src0_spad.size_per_thread = hex_round_up(cws * src0->ne[0] * sizeof(float), VLEN) +
                                          hex_round_up(cws * src0->ne[0] * sizeof(float), VLEN) + VLEN;
        octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
        if (octx->src0_spad.size > octx->ctx->vtcm_size) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }
        octx->src0_spad.data = octx->ctx->vtcm_base;
        octx->src0_spad.src  = NULL;
        work_queue_run(octx->ctx->work_queue, concat_chanfast_f32, &cctx, n_threads);
        return HTP_STATUS_OK;
    }

    if (concat_dim1_contiguous_dma(octx, dim, type_size)) {
        return HTP_STATUS_OK;
    }

    uint32_t n_threads = octx->n_threads;
    struct htp_concat_context cctx;
    cctx.octx = octx;
    cctx.dim = dim;
    cctx.div_ne0 = init_fastdiv_values(dst->ne[0]);
    cctx.div_ne1 = init_fastdiv_values(dst->ne[1]);
    cctx.div_ne2 = init_fastdiv_values(dst->ne[2]);

    void (*worker_func)(unsigned int, unsigned int, void *) = concat_generic;

    const bool rows_ok = src0->nb[0] == type_size && src1->nb[1] == type_size && dst->nb[0] == type_size;

    const uint32_t cat_w = src0->ne[0] + src1->ne[0];
    if (!(octx->ctx->tandem_off & 4) && dim == 0 && type_size == 4 && src0->type == dst->type && src1->type == dst->type &&
        cat_w <= 32 && (32 % cat_w) == 0 && octx->ctx->mdev.count <= 1 &&
        src0->nb[0] == 4 && src0->nb[1] == src0->ne[0] * 4 && src1->nb[1] == 4 &&
        dst->nb[0] == 4 && dst->nb[1] == cat_w * 4 &&
        src0->ne[1] == dst->ne[1] && src1->ne[1] == dst->ne[1] &&
        !htp_tensor_is_extended(src0) && !htp_tensor_is_extended(src1) && !htp_tensor_is_extended(dst)) {
        cctx.nplanes          = dst->ne[2] * dst->ne[3];
        cctx.nrows_per_thread = hex_round_up((dst->ne[1] + n_threads - 1) / n_threads, 32);
        const size_t per = cctx.nrows_per_thread * sizeof(float);
        octx->src0_spad.size_per_thread = hex_round_up(per * src0->ne[0], VLEN) + hex_round_up(per * src1->ne[0], VLEN) +
                                          hex_round_up(per * cat_w, VLEN) + VLEN;
        octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
        if (octx->src0_spad.size <= octx->ctx->vtcm_size) {
            octx->src0_spad.data = octx->ctx->vtcm_base;
            octx->src0_spad.src  = NULL;
            work_queue_run(octx->ctx->work_queue, concat_dim0_gather_f32, &cctx, n_threads);
            return HTP_STATUS_OK;
        }
    }

    if (dim == 0 && is_src1_transposed && !is_src0_transposed && rows_ok) {
        const uint32_t total_rows = dst->ne[1];
        const size_t dst_data_row_size = dst->ne[0] * type_size;
        uint32_t row_start = 0;
        uint32_t nrows     = total_rows;
        if (octx->ctx->mdev.count > 1) {
            uint32_t rows_per_chunk = 0;
            htp_tensor_mdev_rows_per_chunk(dst, type_size, (uint32_t) dst_data_row_size, &rows_per_chunk);
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            row_start = range.start;
            nrows     = range.count;
        }

        if (nrows == 0) {
            return HTP_STATUS_OK;
        }

        cctx.row_start = row_start;
        cctx.nrows     = nrows;
        cctx.nplanes   = dst->ne[2] * dst->ne[3];

        uint32_t block_i = (type_size == 4) ? 32 : 64;

        cctx.nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

        // Allocate VTCM
        uint32_t spad1_stride = block_i * type_size;

        uint32_t src1_ne0_padded = hex_round_up(src1->ne[0], block_i);
        uint32_t spad0_row_bytes = hex_round_up((src0->ne[0] + src1_ne0_padded) * type_size, VLEN);

        octx->src0_spad.size_per_thread = block_i * spad0_row_bytes;
        octx->src1_spad.size_per_thread = src1_ne0_padded * spad1_stride + block_i * VLEN;

        octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
        octx->src1_spad.size = n_threads * octx->src1_spad.size_per_thread;

        if (octx->src0_spad.size + octx->src1_spad.size > octx->ctx->vtcm_size) {
            return HTP_STATUS_VTCM_TOO_SMALL;
        }

        octx->src0_spad.data = octx->ctx->vtcm_base;
        octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size;
        octx->src0_spad.src  = NULL;
        octx->src1_spad.src  = NULL;

        if (type_size == 4) {
            worker_func = concat_2d_f32_transposed;
        } else {
            worker_func = concat_2d_f16_transposed;
        }
    } else {
        if (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
            return HTP_STATUS_NO_SUPPORT;
        }

        const uint32_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
        uint32_t elem_start = 0;
        uint32_t nelems     = total_elements;
        if (octx->ctx->mdev.count > 1) {
            const uint32_t elems_per_chunk = HEX_L2_LINE_SIZE / type_size;
            const bool can_split = htp_tensor_mdev_data_aligned(dst) && htp_tensor_is_contiguous(dst, type_size) && !htp_tensor_is_permuted(dst);
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_elements, can_split ? elems_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
            elem_start = range.start;
            nelems     = range.count;
        }

        if (nelems == 0) {
            return HTP_STATUS_OK;
        }

        cctx.elem_start = elem_start;
        cctx.nelems     = nelems;
    }

    work_queue_run(octx->ctx->work_queue, worker_func, &cctx, n_threads);
    return HTP_STATUS_OK;
}
