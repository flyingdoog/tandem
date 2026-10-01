// Dynamic quantizers that produce tiled activations

static inline void quantize_block_f32_q8_1_tiled(float * restrict x, uint8_t * restrict y_block, uint8_t * restrict rt) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_block % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero = Q6_V_vzero();

    HVX_Vector vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
    HVX_Vector vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
    HVX_Vector vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
    HVX_Vector vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_Vector vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_Vector vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_Vector vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_Vector vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_Vector vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_Vector vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    HVX_Vector vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);
    if (rt) {
        *(HVX_Vector *) rt = vx_i8;
    }

    const HVX_Vector ones = Q6_Vb_vsplat_R(1);
    HVX_Vector v_sums = Q6_Vw_vrmpy_VbVb(vx_i8, ones);
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 4));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 8));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 16));

    const HVX_Vector v_inv127 = hvx_vec_splat_f32(1.0f / 127.0f);
    HVX_Vector vd0_sf = hvx_vec_mul_f32_f32(vmax0_sf, v_inv127);
    HVX_Vector vd1_sf = hvx_vec_mul_f32_f32(vmax1_sf, v_inv127);
    HVX_Vector vd2_sf = hvx_vec_mul_f32_f32(vmax2_sf, v_inv127);
    HVX_Vector vd3_sf = hvx_vec_mul_f32_f32(vmax3_sf, v_inv127);

    HVX_Vector v_sums_sf = Q6_Vsf_equals_Vw(v_sums);
    HVX_Vector voff0_sf = hvx_vec_mul_f32_f32(vd0_sf, v_sums_sf);
    HVX_Vector voff1_sf = hvx_vec_mul_f32_f32(vd1_sf, Q6_V_vror_VR(v_sums_sf, 32));
    HVX_Vector voff2_sf = hvx_vec_mul_f32_f32(vd2_sf, Q6_V_vror_VR(v_sums_sf, 64));
    HVX_Vector voff3_sf = hvx_vec_mul_f32_f32(vd3_sf, Q6_V_vror_VR(v_sums_sf, 96));

    HVX_Vector voff01_hf = hvx_vec_f32_to_f16(voff0_sf, voff1_sf);
    HVX_Vector voff23_hf = hvx_vec_f32_to_f16(voff2_sf, voff3_sf);

    HVX_Vector r_scale[4] = {
        hvx_vec_repl_f16(vd01_hf),
        hvx_vec_repl_f16(Q6_V_vror_VR(vd01_hf, 64)),
        hvx_vec_repl_f16(vd23_hf),
        hvx_vec_repl_f16(Q6_V_vror_VR(vd23_hf, 64)),
    };
    HVX_Vector r_offset[4] = {
        hvx_vec_repl_f16(voff01_hf),
        hvx_vec_repl_f16(Q6_V_vror_VR(voff01_hf, 64)),
        hvx_vec_repl_f16(voff23_hf),
        hvx_vec_repl_f16(Q6_V_vror_VR(voff23_hf, 64)),
    };

    static const uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_Vector v_repl_ctrl = * (const HVX_Vector *) repl;

    for (int b = 0; b < 4; b++) {
        HVX_Vector v_act = Q6_V_vror_VR(vx_i8, b * 32);

        HVX_Vector r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_Vector r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_Vector r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_Vector r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_Vector r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_Vector r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_Vector r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_Vector r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_Vector * restrict dst = (HVX_Vector *) (y_block + b * 1280);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale[b];
        dst[9] = r_offset[b];
    }
}

static inline void quantize_block_f32_q8_0_tiled(float * restrict x, uint8_t * restrict y_block, uint8_t * restrict rt) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_block % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero   = Q6_V_vzero();

    HVX_Vector vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
    HVX_Vector vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
    HVX_Vector vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
    HVX_Vector vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_Vector vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_Vector vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_Vector vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_Vector vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_Vector vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_Vector vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    HVX_Vector vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);
    if (rt) {
        *(HVX_Vector *) rt = vx_i8;
    }

    HVX_VectorPair vp01 = Q6_W_vshuff_VVR(vd01_hf, vd01_hf, -64);
    HVX_VectorPair vp23 = Q6_W_vshuff_VVR(vd23_hf, vd23_hf, -64);

    static const uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_Vector v_repl_ctrl = * (const HVX_Vector *) repl;

    #pragma unroll
    for (int b = 0; b < 4; b++) {
        HVX_Vector v_act = Q6_V_vror_VR(vx_i8, b * 32);
        HVX_Vector r_scale;
        if (b == 0) {
            r_scale = Q6_V_lo_W(vp01);
        } else if (b == 1) {
            r_scale = Q6_V_hi_W(vp01);
        } else if (b == 2) {
            r_scale = Q6_V_lo_W(vp23);
        } else {
            r_scale = Q6_V_hi_W(vp23);
        }

        HVX_Vector r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_Vector r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_Vector r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_Vector r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_Vector r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_Vector r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_Vector r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_Vector r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_Vector * restrict dst = (HVX_Vector *) (y_block + b * 1152);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale;
    }
}

static void quantize_row_f32_q8_0_tiled(float * restrict x, uint8_t * restrict y, uint32_t k, uint8_t * restrict rt) {
    assert(k % 32 == 0);
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (k + qk - 1) / qk;

    for (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block = y + i * 4 * 1152;
        quantize_block_f32_q8_0_tiled(x + i * qk, y_block, rt ? rt + i * qk : NULL);
    }
}

static void quantize_row_f32_q8_1_tiled(float * restrict x, uint8_t * restrict y, uint32_t k, uint8_t * restrict rt) {
    assert(k % 32 == 0);
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (k + qk - 1) / qk;

    for (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block = y + i * 4 * 1280;
        quantize_block_f32_q8_1_tiled(x + i * qk, y_block, rt ? rt + i * qk : NULL);
    }
}

// Dot kernels & helpers that consume tiled activations

static inline HVX_Vector hvx_vec_mul_f16_f16_to_f32_lower32(HVX_Vector v1, HVX_Vector v2) {
#if __HVX_ARCH__ >= 79
    HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    return Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4));
#else
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_Vector hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_Vector lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    return Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -4));
#endif
}

static inline HVX_Vector unpack_and_interleave_4bit(HVX_Vector v_a, HVX_Vector v_b, HVX_Vector mask_h4) {
    HVX_Vector v_W0 = Q6_V_vand_VV(v_a, mask_h4);
    HVX_Vector v_W1 = Q6_Vub_vlsr_VubR(v_a, 4);
    HVX_Vector v_W2 = Q6_V_vand_VV(v_b, mask_h4);
    HVX_Vector v_W3 = Q6_Vub_vlsr_VubR(v_b, 4);

    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_W1, v_W0, -1);
    HVX_VectorPair v23_pair = Q6_W_vshuff_VVR(v_W3, v_W2, -1);
    HVX_VectorPair v0123_pair = Q6_W_vshuff_VVR(Q6_V_lo_W(v23_pair), Q6_V_lo_W(v01_pair), -2);
    return Q6_V_lo_W(v0123_pair);
}

static inline HVX_VectorPair unpack_and_interleave_4bit_x2(HVX_Vector v_src, HVX_Vector mask_h4) {
    HVX_Vector v_lo = Q6_V_vand_VV(v_src, mask_h4);
    HVX_Vector v_hi = Q6_Vub_vlsr_VubR(v_src, 4);
    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_Vector v01_lo = Q6_V_lo_W(v01_pair);
    HVX_Vector v01_hi = Q6_V_hi_W(v01_pair);

    HVX_Vector v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_Vector v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_Vector v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_Vector v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    return Q6_W_vcombine_VV(v_W1, v_W0);
}

static inline HVX_Vector accum_4bit_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

static inline HVX_Vector accum_4bit_32x1_lut(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector mask_h4,
    HVX_Vector lut
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_Vector v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

static inline HVX_VectorPair accum_4bit_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

static inline HVX_VectorPair accum_4bit_32x2_lut(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector mask_h4,
    HVX_Vector lut
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_Vector v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

static inline HVX_Vector accum_q8_0_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act
) {
    HVX_Vector v_sum = Q6_V_vzero();
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        HVX_Vector v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_Vector v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum = Q6_Vw_vrmpyacc_VwVbVb(v_sum, v_W, v_act[g]);
    }
    return v_sum;
}

static inline HVX_VectorPair accum_q8_0_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        HVX_Vector v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_Vector v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W, v_act0[g]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W, v_act1[g]);
    }
    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q5_K: OR 0x10 into every lane of v whose flag j is set in the plane (see HTP_MM_WEIGHT_TILE_SIZE_Q5_K)
static inline HVX_Vector hvx_q5k_or_hibit(HVX_Vector v, HVX_Vector v_plane, int j) {
    HVX_VectorPred q = Q6_Q_vand_VR(v_plane, 0x01010101u << j);
    return Q6_V_vandor_VQR(v, q, 0x10101010);
}

// 5-bit variant: the high bit comes from the plane, see hvx_q5k_or_hibit
static inline HVX_VectorPair unpack_and_interleave_5bit_x2(HVX_Vector v_src, HVX_Vector v_plane, int i, HVX_Vector mask_h4) {
    HVX_Vector v_lo = hvx_q5k_or_hibit(Q6_V_vand_VV(v_src, mask_h4), v_plane, 2 * i);
    HVX_Vector v_hi = hvx_q5k_or_hibit(Q6_Vub_vlsr_VubR(v_src, 4), v_plane, 2 * i + 1);
    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_Vector v01_lo = Q6_V_lo_W(v01_pair);
    HVX_Vector v01_hi = Q6_V_hi_W(v01_pair);

    HVX_Vector v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_Vector v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_Vector v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_Vector v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    return Q6_W_vcombine_VV(v_W1, v_W0);
}

static inline HVX_Vector accum_5bit_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_plane = vptr[5];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

static inline HVX_VectorPair accum_5bit_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_plane = vptr[5];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q6_K weights are stored unsigned (0..63), see HTP_MM_WEIGHT_TILE_SIZE_Q6_K. Unpack k-group g of a tile to signed bytes (q - 32)
static inline HVX_Vector unpack_q6_k_group(const HVX_Vector * restrict vptr, int g, HVX_Vector mask_0f, HVX_Vector mask_03, HVX_Vector i32) {
    HVX_Vector v_lo = (g & 1) ? Q6_Vub_vlsr_VubR(vptr[g >> 1], 4) : Q6_V_vand_VV(vptr[g >> 1], mask_0f);
    HVX_Vector v_hi = (g & 3) ? Q6_Vub_vlsr_VubR(vptr[4 + (g >> 2)], 2 * (g & 3)) : vptr[4 + (g >> 2)];
    HVX_Vector v_q  = Q6_V_vor_VV(v_lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_hi, mask_03), 4));
    return Q6_Vb_vsub_VbVb(v_q, i32);
}

// k 0..15 and k 16..31 of a Q6_K tile have different scales: lo half of the pair sums k 0..15, hi half sums k 16..31
static inline HVX_VectorPair accum_q6_k_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector i32
) {
    HVX_Vector v_sum_lo = Q6_V_vzero();
    HVX_Vector v_sum_hi = Q6_V_vzero();
    HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    for (int g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q6_k_group(vptr, g,     mask_0f, mask_03, i32);
        HVX_Vector v_W_hi = unpack_q6_k_group(vptr, g + 4, mask_0f, mask_03, i32);
        v_sum_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum_lo, v_W_lo, v_act[g]);
        v_sum_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum_hi, v_W_hi, v_act[g + 4]);
    }

    return Q6_W_vcombine_VV(v_sum_hi, v_sum_lo);
}

static inline void accum_q6_k_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector i32,
    HVX_VectorPair * v_sums0,
    HVX_VectorPair * v_sums1
) {
    HVX_Vector v_sum0_lo = Q6_V_vzero();
    HVX_Vector v_sum0_hi = Q6_V_vzero();
    HVX_Vector v_sum1_lo = Q6_V_vzero();
    HVX_Vector v_sum1_hi = Q6_V_vzero();
    HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    for (int g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q6_k_group(vptr, g,     mask_0f, mask_03, i32);
        HVX_Vector v_W_hi = unpack_q6_k_group(vptr, g + 4, mask_0f, mask_03, i32);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// scale the two half sums with the per-row tile scales (v_scale_w = vptr[6]) and the activation scale
static inline HVX_Vector scale_q6_k_32x1(HVX_VectorPair v_sums, HVX_Vector v_scale_w, HVX_Vector v_scale_a) {
    HVX_Vector v_scale_lo = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
    HVX_Vector v_scale_hi = hvx_vec_mul_f16_f16_to_f32_lower32(Q6_V_vror_VR(v_scale_w, 64), v_scale_a);
    HVX_Vector v_lo = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums)), v_scale_lo);
    HVX_Vector v_hi = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums)), v_scale_hi);
    return hvx_vec_add_f32_f32(v_lo, v_hi);
}

static void tiled_vec_dot_q4_0_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act, i8);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_q4_0_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, i8);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_q4_1_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1280);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_sum_a   = v_act[9];

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a);
        HVX_Vector v_offset_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a);

        HVX_Vector v_scaled_dot = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
        HVX_Vector v_sum_scaled = hvx_vec_add_f32_f32(v_scaled_dot, v_offset_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_q4_1_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1280);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1280);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_sum_a_c0   = v_act0[9];
        HVX_Vector v_scale_a_c1 = v_act1[8];
        HVX_Vector v_sum_a_c1   = v_act1[9];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c0);
        HVX_Vector v_offset_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c1);
        HVX_Vector v_offset_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c1);

        HVX_Vector v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c0 = hvx_vec_add_f32_f32(v_scaled_dot_c0, v_offset_comb_c0);

        HVX_Vector v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_add_f32_f32(v_scaled_dot_c1, v_offset_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_q8_0_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 1152);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_q8_0_32x1(vptr, v_act);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[8];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_q8_0_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 1152);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_q8_0_32x2(vptr, v_act0, v_act1);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[8];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_q5_k_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 768);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1280);

        HVX_Vector v_sum = accum_5bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_sum_a   = v_act[9];

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a);
        HVX_Vector v_offset_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a);

        HVX_Vector v_scaled_dot = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
        HVX_Vector v_sum_scaled = hvx_vec_add_f32_f32(v_scaled_dot, v_offset_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_q5_k_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 768);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1280);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1280);

        HVX_VectorPair v_sums = accum_5bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_sum_a_c0   = v_act0[9];
        HVX_Vector v_scale_a_c1 = v_act1[8];
        HVX_Vector v_sum_a_c1   = v_act1[9];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c0);
        HVX_Vector v_offset_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c1);
        HVX_Vector v_offset_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c1);

        HVX_Vector v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c0 = hvx_vec_add_f32_f32(v_scaled_dot_c0, v_offset_comb_c0);

        HVX_Vector v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_add_f32_f32(v_scaled_dot_c1, v_offset_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_q6_k_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 896);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_VectorPair v_sums = accum_q6_k_32x1(vptr, v_act, i32);
        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, scale_q6_k_32x1(v_sums, vptr[6], v_act[8]));
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_q6_k_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 896);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums0, v_sums1;
        accum_q6_k_32x2(vptr, v_act0, v_act1, i32, &v_sums0, &v_sums1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, scale_q6_k_32x1(v_sums0, vptr[6], v_act0[8]));
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, scale_q6_k_32x1(v_sums1, vptr[6], v_act1[8]));
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_iq4nl_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_iq4nl_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_mxfp4_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector v_scale_a_f16 = v_act[8];
        HVX_VectorPair p_scale_a_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_f16);
        HVX_Vector v_scale_a = Q6_V_lo_W(p_scale_a_f32);

        HVX_Vector v_scale_comb = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    v_sum_float = hvx_vec_mul_f32_f32(v_sum_float, hvx_vec_splat_f32(0.5f));

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_mxfp4_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector v_scale_a_c0_f16 = v_act0[8];
        HVX_Vector v_scale_a_c1_f16 = v_act1[8];

        HVX_VectorPair p_scale_a_c0_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_c0_f16);
        HVX_VectorPair p_scale_a_c1_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_c1_f16);

        HVX_Vector v_scale_a_c0 = Q6_V_lo_W(p_scale_a_c0_f32);
        HVX_Vector v_scale_a_c1 = Q6_V_lo_W(p_scale_a_c1_f32);

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    v_sum_float_c0 = hvx_vec_mul_f32_f32(v_sum_float_c0, hvx_vec_splat_f32(0.5f));
    v_sum_float_c1 = hvx_vec_mul_f32_f32(v_sum_float_c1, hvx_vec_splat_f32(0.5f));

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static inline void quantize_f32_q8_0_tiled_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_size,
    size_t dst_row_size
) {
    uint8_t * restrict rt = tmp_data;
    for (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_q8_0_tiled((float *) src_data, dst_data, ne0, rt);
        dst_data += dst_row_size;
        src_data += src_row_size;
        rt = rt ? rt + hex_round_up(ne0, QK_Q8_0_TILED) : NULL;
    }
}

static inline void quantize_f32_q8_1_tiled_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_size,
    size_t dst_row_size
) {
    uint8_t * restrict rt = tmp_data;
    for (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_q8_1_tiled((float *) src_data, dst_data, ne0, rt);
        dst_data += dst_row_size;
        src_data += src_row_size;
        rt = rt ? rt + hex_round_up(ne0, QK_Q8_0_TILED) : NULL;
    }
}

static inline void quantize_f32_q8_0_tiled_block_kernel(
    const float * restrict src,
    uint8_t * restrict dst,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_last,
    size_t src_row_size,
    size_t dst_row_size,
    uint32_t r,
    uint32_t c
) {
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne0 + qk - 1) / qk;

    for (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        const float * restrict src_ptr = (const float *) ((const uint8_t *) src + r * src_row_size + c * qk * sizeof(float));
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 1152;
        uint8_t * restrict rt_ptr  = tmp_data ? tmp_data + r * hex_round_up(ne0, QK_Q8_0_TILED) + c * qk : NULL;

        quantize_block_f32_q8_0_tiled((float *) src_ptr, dst_ptr, rt_ptr);

        c++;
        if (c == nb) {
            c = 0;
            r++;
        }
    }
}

static inline void quantize_f32_q8_1_tiled_block_kernel(
    const float * restrict src,
    uint8_t * restrict dst,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_last,
    size_t src_row_size,
    size_t dst_row_size,
    uint32_t r,
    uint32_t c
) {
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne0 + qk - 1) / qk;

    for (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        const float * restrict src_ptr = (const float *) ((const uint8_t *) src + r * src_row_size + c * qk * sizeof(float));
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 1280;
        uint8_t * restrict rt_ptr  = tmp_data ? tmp_data + r * hex_round_up(ne0, QK_Q8_0_TILED) + c * qk : NULL;

        quantize_block_f32_q8_1_tiled((float *) src_ptr, dst_ptr, rt_ptr);

        c++;
        if (c == nb) {
            c = 0;
            r++;
        }
    }
}

// Small-batch "rt" kernels (1..8 activation columns).
// The q8 quantizers also write the plain int8 activations (act_rt, normal cached memory) when requested.
// The kernels read 4 activation bytes at a time into a scalar register and use vrmpy(Vu.ub, Rt.b):
// no replicated activation vectors are loaded, and one weight tile unpack is shared by all columns.
// Scalar loads from VTCM are slow, so act_rt must not live in VTCM. Block scales still come from the tiled VTCM copy.
// vrmpy(Vu.ub, Rt.b) needs unsigned weights: the q4_0 (8) and q6_k (32) biases are removed with activation block sums.

#define HTP_MM_RT_MAX_COLS 8

static inline void rt_store_32xn(uint32_t nc, float * restrict * s, const HVX_Vector * v, uint32_t valid_rows, const float * restrict * sz) {
    for (uint32_t c = 0; c < nc; c++) {
        HVX_Vector r = sz[c] ? hvx_vec_add_f32_f32(v[c], hvx_vmemu(sz[c])) : v[c];
        hvx_vec_store_u(s[c], valid_rows * sizeof(float), r);
    }
}

// sum of 16 signed bytes held in 4 words
static inline int32_t rt_sum_i8x16(const int32_t * a) {
    const int64_t ones = 0x0101010101010101LL;
    int64_t p = Q6_P_vrmpybsu_PP(Q6_P_combine_RR(a[1], a[0]), ones);
    p = Q6_P_vrmpybsuacc_PP(p, Q6_P_combine_RR(a[3], a[2]), ones);
    return (int32_t) p + (int32_t) (p >> 32);
}

static inline void rt_load_act(const uint8_t * restrict yr, int32_t * aw) {
    const int64_t * restrict p = (const int64_t *) yr;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int64_t v = p[i];
        aw[2 * i + 0] = (int32_t) v;
        aw[2 * i + 1] = (int32_t) (v >> 32);
    }
}

// q4_1, q4_k (repacked as q4_1) and q5_k: scale + offset epilogue
static inline HVX_Vector rt_scale_offset_32x1(HVX_Vector v_sum, HVX_Vector v_scale, HVX_Vector v_offset, const HVX_Vector * restrict v_act) {
    HVX_Vector v_scale_comb  = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_act[8]);
    HVX_Vector v_offset_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_act[9]);
    return hvx_vec_add_f32_f32(hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(v_sum), v_scale_comb), v_offset_comb);
}

// Q6_K group unpack without the -32 bias (0..63)
static inline HVX_Vector unpack_q6_k_group_u(const HVX_Vector * restrict vptr, int g, HVX_Vector mask_0f, HVX_Vector mask_03) {
    HVX_Vector v_lo = (g & 1) ? Q6_Vub_vlsr_VubR(vptr[g >> 1], 4) : Q6_V_vand_VV(vptr[g >> 1], mask_0f);
    HVX_Vector v_hi = (g & 3) ? Q6_Vub_vlsr_VubR(vptr[4 + (g >> 2)], 2 * (g & 3)) : vptr[4 + (g >> 2)];
    return Q6_V_vor_VV(v_lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_hi, mask_03), 4));
}

// q4_1 / q4_k (tile 640) and q5_k (tile 768) with q8_1 activations (1280 per k tile)
static inline __attribute__((always_inline)) void rt_dot_q4_1_nc(const uint32_t n, const uint32_t nc, const int is_q5,
        float * restrict * s, const void * restrict vx, const uint8_t * restrict * vy, const uint8_t * restrict * vyr,
        uint32_t valid_rows, const float * restrict * sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint32_t tile_size = is_q5 ? 768 : 640;
    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    HVX_Vector v_acc[HTP_MM_RT_MAX_COLS];
    for (uint32_t c = 0; c < nc; c++) {
        v_acc[c] = Q6_V_vzero();
    }

    const uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * tile_size);

        HVX_Vector v_w[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            HVX_VectorPair p = is_q5 ? unpack_and_interleave_5bit_x2(vptr[i], vptr[5], i, mask_h4) : unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
            v_w[2 * i + 0] = Q6_V_lo_W(p);
            v_w[2 * i + 1] = Q6_V_hi_W(p);
        }
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(vptr[4], vptr[4], -2);

        #pragma unroll
        for (uint32_t c = 0; c < nc; c++) {
            int32_t aw[8];
            rt_load_act(vyr[c] + kt * 32, aw);
            HVX_Vector v_sum = Q6_V_vzero();
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                v_sum = Q6_Vw_vrmpyacc_VwVubRb(v_sum, v_w[g], aw[g]);
            }
            const HVX_Vector * restrict v_act = (const HVX_Vector *) (vy[c] + kt * 1280);
            v_acc[c] = hvx_vec_add_f32_f32(v_acc[c], rt_scale_offset_32x1(v_sum, Q6_V_lo_W(p_deal), Q6_V_hi_W(p_deal), v_act));
        }
    }

    rt_store_32xn(nc, s, v_acc, valid_rows, sz);
}

// q4_0 (tile 640) with q8_0 activations (1152 per k tile); weights 0..15, bias 8 removed with the block sum
static inline __attribute__((always_inline)) void rt_dot_q4_0_nc(const uint32_t n, const uint32_t nc,
        float * restrict * s, const void * restrict vx, const uint8_t * restrict * vy, const uint8_t * restrict * vyr,
        uint32_t valid_rows, const float * restrict * sz) {
    const uint8_t * restrict tile_ptr = vx;
    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    HVX_Vector v_acc[HTP_MM_RT_MAX_COLS];
    for (uint32_t c = 0; c < nc; c++) {
        v_acc[c] = Q6_V_vzero();
    }

    const uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);

        HVX_Vector v_w[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            HVX_VectorPair p = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
            v_w[2 * i + 0] = Q6_V_lo_W(p);
            v_w[2 * i + 1] = Q6_V_hi_W(p);
        }
        HVX_Vector v_scale_w = vptr[4];

        #pragma unroll
        for (uint32_t c = 0; c < nc; c++) {
            int32_t aw[8];
            rt_load_act(vyr[c] + kt * 32, aw);
            HVX_Vector v_sum = Q6_V_vzero();
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                v_sum = Q6_Vw_vrmpyacc_VwVubRb(v_sum, v_w[g], aw[g]);
            }
            const int32_t sum_a = rt_sum_i8x16(aw) + rt_sum_i8x16(aw + 4);
            v_sum = Q6_Vw_vsub_VwVw(v_sum, Q6_V_vsplat_R(8 * sum_a));

            const HVX_Vector * restrict v_act = (const HVX_Vector *) (vy[c] + kt * 1152);
            HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_act[8]);
            v_acc[c] = hvx_vec_add_f32_f32(v_acc[c], hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(v_sum), v_scale_comb));
        }
    }

    rt_store_32xn(nc, s, v_acc, valid_rows, sz);
}

// q6_k (tile 896) with q8_0 activations; k 0..15 and k 16..31 have their own scale, bias 32 removed per half
static inline __attribute__((always_inline)) void rt_dot_q6_k_nc(const uint32_t n, const uint32_t nc,
        float * restrict * s, const void * restrict vx, const uint8_t * restrict * vy, const uint8_t * restrict * vyr,
        uint32_t valid_rows, const float * restrict * sz) {
    const uint8_t * restrict tile_ptr = vx;
    const HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    const HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    HVX_Vector v_acc[HTP_MM_RT_MAX_COLS];
    for (uint32_t c = 0; c < nc; c++) {
        v_acc[c] = Q6_V_vzero();
    }

    const uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 896);

        HVX_Vector v_w[8];
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            v_w[g] = unpack_q6_k_group_u(vptr, g, mask_0f, mask_03);
        }

        #pragma unroll
        for (uint32_t c = 0; c < nc; c++) {
            int32_t aw[8];
            rt_load_act(vyr[c] + kt * 32, aw);
            HVX_Vector v_lo = Q6_V_vzero();
            HVX_Vector v_hi = Q6_V_vzero();
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                v_lo = Q6_Vw_vrmpyacc_VwVubRb(v_lo, v_w[g],     aw[g]);
                v_hi = Q6_Vw_vrmpyacc_VwVubRb(v_hi, v_w[g + 4], aw[g + 4]);
            }
            v_lo = Q6_Vw_vsub_VwVw(v_lo, Q6_V_vsplat_R(32 * rt_sum_i8x16(aw)));
            v_hi = Q6_Vw_vsub_VwVw(v_hi, Q6_V_vsplat_R(32 * rt_sum_i8x16(aw + 4)));

            const HVX_Vector * restrict v_act = (const HVX_Vector *) (vy[c] + kt * 1152);
            v_acc[c] = hvx_vec_add_f32_f32(v_acc[c], scale_q6_k_32x1(Q6_W_vcombine_VV(v_hi, v_lo), vptr[6], v_act[8]));
        }
    }

    rt_store_32xn(nc, s, v_acc, valid_rows, sz);
}

#define RT_DOT_DISPATCH(NAME, CALL)                                                                                   \
static void NAME(uint32_t n, uint32_t nc, float * restrict * s, const void * restrict vx,                             \
                 const uint8_t * restrict * vy, const uint8_t * restrict * vyr, uint32_t valid_rows,                  \
                 const float * restrict * sz) {                                                                       \
    switch (nc) {                                                                                                     \
        case 8:  CALL(8); break;                                                                                      \
        case 7:  CALL(7); break;                                                                                      \
        case 6:  CALL(6); break;                                                                                      \
        case 5:  CALL(5); break;                                                                                      \
        case 4:  CALL(4); break;                                                                                      \
        case 3:  CALL(3); break;                                                                                      \
        case 2:  CALL(2); break;                                                                                      \
        default: CALL(1); break;                                                                                      \
    }                                                                                                                 \
}

#define RT_CALL_Q4_1(NC) rt_dot_q4_1_nc(n, NC, 0, s, vx, vy, vyr, valid_rows, sz)
#define RT_CALL_Q5_K(NC) rt_dot_q4_1_nc(n, NC, 1, s, vx, vy, vyr, valid_rows, sz)
#define RT_CALL_Q4_0(NC) rt_dot_q4_0_nc(n, NC, s, vx, vy, vyr, valid_rows, sz)
#define RT_CALL_Q6_K(NC) rt_dot_q6_k_nc(n, NC, s, vx, vy, vyr, valid_rows, sz)

RT_DOT_DISPATCH(rt_dot_q4_1, RT_CALL_Q4_1)
RT_DOT_DISPATCH(rt_dot_q5_k, RT_CALL_Q5_K)
RT_DOT_DISPATCH(rt_dot_q4_0, RT_CALL_Q4_0)
RT_DOT_DISPATCH(rt_dot_q6_k, RT_CALL_Q6_K)

// number of columns for the next rt kernel call
static inline uint32_t rt_cols(uint32_t rem) {
    return rem >= HTP_MM_RT_MAX_COLS ? HTP_MM_RT_MAX_COLS : rem;
}

