/*
 * RISC-V translation routines for the RVV Standard Extension.
 *
 * Copyright (c) 2020 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2 or later, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "tcg/tcg-op-gvec.h"
#include "tcg/tcg-gvec-desc.h"
#include "internals.h"

static bool do_vsetvl_7(DisasContext *ctx, int rd, int rs1, TCGv s2)
{
    TCGv s1, dst;

    if (!has_ext(ctx, RVV)) {
        return false;
    }

    dst = dest_gpr(ctx, rd);

    /* Using x0 as the rs1 register specifier, encodes an infinite AVL */
    if (rs1 == 0) {
        /* As the mask is at least one bit, RV_VLEN_MAX is >= VLMAX */
        s1 = tcg_constant_tl(RV_VLEN_MAX);
    } else {
        s1 = get_gpr(ctx, rs1, EXT_ZERO);
    }
    gen_helper_vsetvl_7(dst, cpu_env, s1, s2, tcg_constant_tl(get_xlen(ctx)));
    gen_set_gpr(ctx, rd, dst);

    tcg_gen_movi_tl(cpu_pc, ctx->pc_succ_insn);
    if (gen_tb_trace()) {
        csky_trace_tb_exit(0x2, ctx->pc_succ_insn - ctx->base.pc_first);
    }
    lookup_and_goto_ptr(ctx);
    ctx->base.is_jmp = DISAS_NORETURN;
    return true;
}

static bool trans_vsetvl_7(DisasContext *ctx, arg_vsetvl_7 *a)
{
    TCGv s2 = get_gpr(ctx, a->rs2, EXT_ZERO);
    return do_vsetvl_7(ctx, a->rd, a->rs1, s2);
}

static bool trans_vsetvli_7(DisasContext *ctx, arg_vsetvli_7 *a)
{
    TCGv s2 = tcg_constant_tl(a->zimm);
    return do_vsetvl_7(ctx, a->rd, a->rs1, s2);
}

/* check functions */

/*
 * There are two rules check here.
 *
 * 1. Vector register numbers are multiples of LMUL. (Section 3.2)
 *
 * 2. For all widening instructions, the destination LMUL value must also be
 *    a supported LMUL value. (Section 11.2)
 */
static bool vext_check_reg(DisasContext *s, uint32_t reg, bool widen)
{
    /*
     * The destination vector register group results are arranged as if both
     * SEW and LMUL were at twice their current settings. (Section 11.2).
     */
    int legal = widen ? 2 << s->lmul : 1 << s->lmul;

    return !((s->lmul == 0x3 && widen) || (reg % legal));
}

/*
 * There are two rules check here.
 *
 * 1. The destination vector register group for a masked vector instruction can
 *    only overlap the source mask register (v0) when LMUL=1. (Section 5.3)
 *
 * 2. In widen instructions and some other insturctions, like vslideup.vx,
 *    there is no need to check whether LMUL=1.
 */
static bool vext_check_overlap_mask(DisasContext *s, uint32_t vd, bool vm,
    bool force)
{
    return (vm != 0 || vd != 0) || (!force && (s->lmul == 0));
}

/* The LMUL setting must be such that LMUL * NFIELDS <= 8. (Section 7.8) */
static bool vext_check_nf(DisasContext *s, uint32_t nf)
{
    return (1 << s->lmul) * nf <= 8;
}

/*
 * The destination vector register group cannot overlap a source vector register
 * group of a different element width. (Section 11.2)
 */
static inline bool vext_check_overlap_group(int rd, int dlen, int rs, int slen)
{
    return ((rd >= rs + slen) || (rs >= rd + dlen));
}
/* common translation macro */
#undef GEN_VEXT_TRANS
#define GEN_VEXT_TRANS(NAME, SEQ, ARGTYPE, OP, CHECK)      \
static bool trans_##NAME(DisasContext *s, arg_##ARGTYPE *a)\
{                                                          \
    if (CHECK(s, a)) {                                     \
        return OP(s, a, SEQ);                              \
    }                                                      \
    return false;                                          \
}

/*
 *** unit stride load and store
 */
static bool ldst_us_trans_7(uint32_t vd, uint32_t rs1, uint32_t data,
                            gen_helper_ldst_us *fn, DisasContext *s)
{
    TCGv_ptr dest, mask;
    TCGv base;
    TCGv_i32 desc;

    TCGLabel *over = gen_new_label();
    tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    base = get_gpr(s, rs1, EXT_NONE);

    /*
     * As simd_desc supports at most 256 bytes, and in this implementation,
     * the max vector group length is 2048 bytes. So split it into two parts.
     *
     * The first part is vlen in bytes, encoded in maxsz of simd_desc.
     * The second part is lmul, encoded in data of simd_desc.
     */
    desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));

    tcg_gen_addi_ptr(dest, cpu_env, vreg_ofs(s, vd));
    tcg_gen_addi_ptr(mask, cpu_env, vreg_ofs(s, 0));

    fn(dest, mask, base, cpu_env, desc);

    gen_set_label(over);
    return true;
}

static bool ld_us_op_7(DisasContext *s, arg_r2nfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_us *fn;
    static gen_helper_ldst_us * const fns[2][7][4] = {
        /* masked unit stride load */
        { { gen_helper_vlb_v_b_mask,  gen_helper_vlb_v_h_mask,
            gen_helper_vlb_v_w_mask,  gen_helper_vlb_v_d_mask },
          { NULL,                     gen_helper_vlh_v_h_mask,
            gen_helper_vlh_v_w_mask,  gen_helper_vlh_v_d_mask },
          { NULL,                     NULL,
            gen_helper_vlw_v_w_mask,  gen_helper_vlw_v_d_mask },
          { gen_helper_vle_v_b_mask,  gen_helper_vle_v_h_mask,
            gen_helper_vle_v_w_mask,  gen_helper_vle_v_d_mask },
          { gen_helper_vlbu_v_b_mask, gen_helper_vlbu_v_h_mask,
            gen_helper_vlbu_v_w_mask, gen_helper_vlbu_v_d_mask },
          { NULL,                     gen_helper_vlhu_v_h_mask,
            gen_helper_vlhu_v_w_mask, gen_helper_vlhu_v_d_mask },
          { NULL,                     NULL,
            gen_helper_vlwu_v_w_mask, gen_helper_vlwu_v_d_mask } },
        /* unmasked unit stride load */
        { { gen_helper_vlb_v_b,  gen_helper_vlb_v_h,
            gen_helper_vlb_v_w,  gen_helper_vlb_v_d },
          { NULL,                gen_helper_vlh_v_h,
            gen_helper_vlh_v_w,  gen_helper_vlh_v_d },
          { NULL,                NULL,
            gen_helper_vlw_v_w,  gen_helper_vlw_v_d },
          { gen_helper_vle_v_b,  gen_helper_vle_v_h,
            gen_helper_vle_v_w,  gen_helper_vle_v_d },
          { gen_helper_vlbu_v_b, gen_helper_vlbu_v_h,
            gen_helper_vlbu_v_w, gen_helper_vlbu_v_d },
          { NULL,                gen_helper_vlhu_v_h,
            gen_helper_vlhu_v_w, gen_helper_vlhu_v_d },
          { NULL,                NULL,
            gen_helper_vlwu_v_w, gen_helper_vlwu_v_d } }
    };

    fn =  fns[a->vm][seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_7, VM, a->vm);
    data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_7, NF, a->nf);
    data = FIELD_DP32(data, VDATA_7, OL, get_xl(s));
    return ldst_us_trans_7(a->rd, a->rs1, data, fn, s);
}

static bool ld_us_check_7(DisasContext *s, arg_r2nfvm* a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_nf(s, a->nf));
}

GEN_VEXT_TRANS(vlb_v, 0, r2nfvm, ld_us_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vlh_v, 1, r2nfvm, ld_us_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vlw_v, 2, r2nfvm, ld_us_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vle_v, 3, r2nfvm, ld_us_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vlbu_v, 4, r2nfvm, ld_us_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vlhu_v, 5, r2nfvm, ld_us_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vlwu_v, 6, r2nfvm, ld_us_op_7, ld_us_check_7)

static bool st_us_op_7(DisasContext *s, arg_r2nfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_us *fn;
    static gen_helper_ldst_us * const fns[2][4][4] = {
        /* masked unit stride load and store */
        { { gen_helper_vsb_v_b_mask,  gen_helper_vsb_v_h_mask,
            gen_helper_vsb_v_w_mask,  gen_helper_vsb_v_d_mask },
          { NULL,                     gen_helper_vsh_v_h_mask,
            gen_helper_vsh_v_w_mask,  gen_helper_vsh_v_d_mask },
          { NULL,                     NULL,
            gen_helper_vsw_v_w_mask,  gen_helper_vsw_v_d_mask },
          { gen_helper_vse_v_b_mask,  gen_helper_vse_v_h_mask,
            gen_helper_vse_v_w_mask,  gen_helper_vse_v_d_mask } },
        /* unmasked unit stride store */
        { { gen_helper_vsb_v_b,  gen_helper_vsb_v_h,
            gen_helper_vsb_v_w,  gen_helper_vsb_v_d },
          { NULL,                gen_helper_vsh_v_h,
            gen_helper_vsh_v_w,  gen_helper_vsh_v_d },
          { NULL,                NULL,
            gen_helper_vsw_v_w,  gen_helper_vsw_v_d },
          { gen_helper_vse_v_b,  gen_helper_vse_v_h,
            gen_helper_vse_v_w,  gen_helper_vse_v_d } }
    };

    fn =  fns[a->vm][seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_7, VM, a->vm);
    data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_7, NF, a->nf);
    data = FIELD_DP32(data, VDATA_7, OL, get_xl(s));
    return ldst_us_trans_7(a->rd, a->rs1, data, fn, s);
}

static bool st_us_check_7(DisasContext *s, arg_r2nfvm* a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_nf(s, a->nf));
}

GEN_VEXT_TRANS(vsb_v, 0, r2nfvm, st_us_op_7, st_us_check_7)
GEN_VEXT_TRANS(vsh_v, 1, r2nfvm, st_us_op_7, st_us_check_7)
GEN_VEXT_TRANS(vsw_v, 2, r2nfvm, st_us_op_7, st_us_check_7)
GEN_VEXT_TRANS(vse_v, 3, r2nfvm, st_us_op_7, st_us_check_7)

/*
 *** stride load and store
 */
typedef void gen_helper_ldst_stride_7(TCGv_ptr, TCGv_ptr, TCGv,
                                    TCGv, TCGv_env, TCGv_i32);

static bool ldst_stride_trans_7(uint32_t vd, uint32_t rs1, uint32_t rs2,
                                uint32_t data, gen_helper_ldst_stride_7 *fn,
                                DisasContext *s)
{
    TCGv_ptr dest, mask;
    TCGv base, stride;
    TCGv_i32 desc;

    TCGLabel *over = gen_new_label();
    tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    base = get_gpr(s, rs1, EXT_NONE);
    stride = get_gpr(s, rs2, EXT_NONE);
    desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));

    tcg_gen_addi_ptr(dest, cpu_env, vreg_ofs(s, vd));
    tcg_gen_addi_ptr(mask, cpu_env, vreg_ofs(s, 0));

    fn(dest, mask, base, stride, cpu_env, desc);

    gen_set_label(over);
    return true;
}

static bool ld_stride_op_7(DisasContext *s, arg_rnfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_stride_7 *fn;
    static gen_helper_ldst_stride_7 * const fns[7][4] = {
        { gen_helper_vlsb_v_b,  gen_helper_vlsb_v_h,
          gen_helper_vlsb_v_w,  gen_helper_vlsb_v_d },
        { NULL,                 gen_helper_vlsh_v_h,
          gen_helper_vlsh_v_w,  gen_helper_vlsh_v_d },
        { NULL,                 NULL,
          gen_helper_vlsw_v_w,  gen_helper_vlsw_v_d },
        { gen_helper_vlse_v_b,  gen_helper_vlse_v_h,
          gen_helper_vlse_v_w,  gen_helper_vlse_v_d },
        { gen_helper_vlsbu_v_b, gen_helper_vlsbu_v_h,
          gen_helper_vlsbu_v_w, gen_helper_vlsbu_v_d },
        { NULL,                 gen_helper_vlshu_v_h,
          gen_helper_vlshu_v_w, gen_helper_vlshu_v_d },
        { NULL,                 NULL,
          gen_helper_vlswu_v_w, gen_helper_vlswu_v_d },
    };

    fn =  fns[seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_7, VM, a->vm);
    data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_7, NF, a->nf);
    data = FIELD_DP32(data, VDATA_7, OL, get_xl(s));
    return ldst_stride_trans_7(a->rd, a->rs1, a->rs2, data, fn, s);
}

static bool ld_stride_check_7(DisasContext *s, arg_rnfvm* a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_nf(s, a->nf));
}

GEN_VEXT_TRANS(vlsb_v, 0, rnfvm, ld_stride_op_7, ld_stride_check_7)
GEN_VEXT_TRANS(vlsh_v, 1, rnfvm, ld_stride_op_7, ld_stride_check_7)
GEN_VEXT_TRANS(vlsw_v, 2, rnfvm, ld_stride_op_7, ld_stride_check_7)
GEN_VEXT_TRANS(vlse_v, 3, rnfvm, ld_stride_op_7, ld_stride_check_7)
GEN_VEXT_TRANS(vlsbu_v, 4, rnfvm, ld_stride_op_7, ld_stride_check_7)
GEN_VEXT_TRANS(vlshu_v, 5, rnfvm, ld_stride_op_7, ld_stride_check_7)
GEN_VEXT_TRANS(vlswu_v, 6, rnfvm, ld_stride_op_7, ld_stride_check_7)

static bool st_stride_op_7(DisasContext *s, arg_rnfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_stride_7 *fn;
    static gen_helper_ldst_stride_7 * const fns[4][4] = {
        /* masked stride store */
        { gen_helper_vssb_v_b,  gen_helper_vssb_v_h,
          gen_helper_vssb_v_w,  gen_helper_vssb_v_d },
        { NULL,                 gen_helper_vssh_v_h,
          gen_helper_vssh_v_w,  gen_helper_vssh_v_d },
        { NULL,                 NULL,
          gen_helper_vssw_v_w,  gen_helper_vssw_v_d },
        { gen_helper_vsse_v_b,  gen_helper_vsse_v_h,
          gen_helper_vsse_v_w,  gen_helper_vsse_v_d }
    };

    data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_7, VM, a->vm);
    data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_7, NF, a->nf);
    data = FIELD_DP32(data, VDATA_7, OL, get_xl(s));
    fn =  fns[seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    return ldst_stride_trans_7(a->rd, a->rs1, a->rs2, data, fn, s);
}

static bool st_stride_check_7(DisasContext *s, arg_rnfvm* a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_nf(s, a->nf));
}

GEN_VEXT_TRANS(vssb_v, 0, rnfvm, st_stride_op_7, st_stride_check_7)
GEN_VEXT_TRANS(vssh_v, 1, rnfvm, st_stride_op_7, st_stride_check_7)
GEN_VEXT_TRANS(vssw_v, 2, rnfvm, st_stride_op_7, st_stride_check_7)
GEN_VEXT_TRANS(vsse_v, 3, rnfvm, st_stride_op_7, st_stride_check_7)

/*
 *** index load and store
 */
static bool ldst_index_trans_7(uint32_t vd, uint32_t rs1, uint32_t vs2,
                             uint32_t data, gen_helper_ldst_index *fn,
                             DisasContext *s)
{
    TCGv_ptr dest, mask, index;
    TCGv base;
    TCGv_i32 desc;

    TCGLabel *over = gen_new_label();
    tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    index = tcg_temp_new_ptr();
    base = get_gpr(s, rs1, EXT_NONE);
    desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));

    tcg_gen_addi_ptr(dest, cpu_env, vreg_ofs(s, vd));
    tcg_gen_addi_ptr(index, cpu_env, vreg_ofs(s, vs2));
    tcg_gen_addi_ptr(mask, cpu_env, vreg_ofs(s, 0));

    fn(dest, mask, base, index, cpu_env, desc);

    gen_set_label(over);
    return true;
}

static bool ld_index_op_7(DisasContext *s, arg_rnfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_index *fn;
    static gen_helper_ldst_index * const fns[7][4] = {
        { gen_helper_vlxb_v_b,  gen_helper_vlxb_v_h,
          gen_helper_vlxb_v_w,  gen_helper_vlxb_v_d },
        { NULL,                 gen_helper_vlxh_v_h,
          gen_helper_vlxh_v_w,  gen_helper_vlxh_v_d },
        { NULL,                 NULL,
          gen_helper_vlxw_v_w,  gen_helper_vlxw_v_d },
        { gen_helper_vlxe_v_b,  gen_helper_vlxe_v_h,
          gen_helper_vlxe_v_w,  gen_helper_vlxe_v_d },
        { gen_helper_vlxbu_v_b, gen_helper_vlxbu_v_h,
          gen_helper_vlxbu_v_w, gen_helper_vlxbu_v_d },
        { NULL,                 gen_helper_vlxhu_v_h,
          gen_helper_vlxhu_v_w, gen_helper_vlxhu_v_d },
        { NULL,                 NULL,
          gen_helper_vlxwu_v_w, gen_helper_vlxwu_v_d },
    };

    fn =  fns[seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_7, VM, a->vm);
    data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_7, NF, a->nf);
    data = FIELD_DP32(data, VDATA_7, OL, get_xl(s));
    return ldst_index_trans_7(a->rd, a->rs1, a->rs2, data, fn, s);
}

/*
 * For vector indexed segment loads, the destination vector register
 * groups cannot overlap the source vector register group (specified by
 * `vs2`), else an illegal instruction exception is raised.
 */
static bool ld_index_check_7(DisasContext *s, arg_rnfvm* a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_nf(s, a->nf) &&
            ((a->nf == 1) ||
             vext_check_overlap_group(a->rd, a->nf << s->lmul,
                                      a->rs2, 1 << s->lmul)));
}

GEN_VEXT_TRANS(vlxb_v, 0, rnfvm, ld_index_op_7, ld_index_check_7)
GEN_VEXT_TRANS(vlxh_v, 1, rnfvm, ld_index_op_7, ld_index_check_7)
GEN_VEXT_TRANS(vlxw_v, 2, rnfvm, ld_index_op_7, ld_index_check_7)
GEN_VEXT_TRANS(vlxe_v, 3, rnfvm, ld_index_op_7, ld_index_check_7)
GEN_VEXT_TRANS(vlxbu_v, 4, rnfvm, ld_index_op_7, ld_index_check_7)
GEN_VEXT_TRANS(vlxhu_v, 5, rnfvm, ld_index_op_7, ld_index_check_7)
GEN_VEXT_TRANS(vlxwu_v, 6, rnfvm, ld_index_op_7, ld_index_check_7)

static bool st_index_op_7(DisasContext *s, arg_rnfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_index *fn;
    static gen_helper_ldst_index * const fns[4][4] = {
        { gen_helper_vsxb_v_b,  gen_helper_vsxb_v_h,
          gen_helper_vsxb_v_w,  gen_helper_vsxb_v_d },
        { NULL,                 gen_helper_vsxh_v_h,
          gen_helper_vsxh_v_w,  gen_helper_vsxh_v_d },
        { NULL,                 NULL,
          gen_helper_vsxw_v_w,  gen_helper_vsxw_v_d },
        { gen_helper_vsxe_v_b,  gen_helper_vsxe_v_h,
          gen_helper_vsxe_v_w,  gen_helper_vsxe_v_d }
    };

    fn =  fns[seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_7, VM, a->vm);
    data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_7, NF, a->nf);
    data = FIELD_DP32(data, VDATA_7, OL, get_xl(s));
    return ldst_index_trans_7(a->rd, a->rs1, a->rs2, data, fn, s);
}

static bool st_index_check_7(DisasContext *s, arg_rnfvm* a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_nf(s, a->nf));
}

GEN_VEXT_TRANS(vsxb_v, 0, rnfvm, st_index_op_7, st_index_check_7)
GEN_VEXT_TRANS(vsxh_v, 1, rnfvm, st_index_op_7, st_index_check_7)
GEN_VEXT_TRANS(vsxw_v, 2, rnfvm, st_index_op_7, st_index_check_7)
GEN_VEXT_TRANS(vsxe_v, 3, rnfvm, st_index_op_7, st_index_check_7)

/*
 *** unit stride fault-only-first load
 */
static bool ldff_trans_7(uint32_t vd, uint32_t rs1, uint32_t data,
                       gen_helper_ldst_us *fn, DisasContext *s)
{
    TCGv_ptr dest, mask;
    TCGv base;
    TCGv_i32 desc;

    TCGLabel *over = gen_new_label();
    tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    base = get_gpr(s, rs1, EXT_NONE);
    desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));

    tcg_gen_addi_ptr(dest, cpu_env, vreg_ofs(s, vd));
    tcg_gen_addi_ptr(mask, cpu_env, vreg_ofs(s, 0));

    fn(dest, mask, base, cpu_env, desc);

    gen_set_label(over);
    return true;
}

static bool ldff_op_7(DisasContext *s, arg_r2nfvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_ldst_us *fn;
    static gen_helper_ldst_us * const fns[7][4] = {
        { gen_helper_vlbff_v_b,  gen_helper_vlbff_v_h,
          gen_helper_vlbff_v_w,  gen_helper_vlbff_v_d },
        { NULL,                  gen_helper_vlhff_v_h,
          gen_helper_vlhff_v_w,  gen_helper_vlhff_v_d },
        { NULL,                  NULL,
          gen_helper_vlwff_v_w,  gen_helper_vlwff_v_d },
        { gen_helper_vleff_v_b,  gen_helper_vleff_v_h,
          gen_helper_vleff_v_w,  gen_helper_vleff_v_d },
        { gen_helper_vlbuff_v_b, gen_helper_vlbuff_v_h,
          gen_helper_vlbuff_v_w, gen_helper_vlbuff_v_d },
        { NULL,                  gen_helper_vlhuff_v_h,
          gen_helper_vlhuff_v_w, gen_helper_vlhuff_v_d },
        { NULL,                  NULL,
          gen_helper_vlwuff_v_w, gen_helper_vlwuff_v_d }
    };

    fn =  fns[seq][s->sew];
    if (fn == NULL) {
        return false;
    }

    data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_7, VM, a->vm);
    data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_7, NF, a->nf);
    data = FIELD_DP32(data, VDATA_7, OL, get_xl(s));
    return ldff_trans_7(a->rd, a->rs1, data, fn, s);
}

GEN_VEXT_TRANS(vlbff_v, 0, r2nfvm, ldff_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vlhff_v, 1, r2nfvm, ldff_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vlwff_v, 2, r2nfvm, ldff_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vleff_v, 3, r2nfvm, ldff_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vlbuff_v, 4, r2nfvm, ldff_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vlhuff_v, 5, r2nfvm, ldff_op_7, ld_us_check_7)
GEN_VEXT_TRANS(vlwuff_v, 6, r2nfvm, ldff_op_7, ld_us_check_7)

/*
 *** vector atomic operation
 */
typedef void gen_helper_amo(TCGv_ptr, TCGv_ptr, TCGv, TCGv_ptr,
                            TCGv_env, TCGv_i32);

static bool amo_trans_7(uint32_t vd, uint32_t rs1, uint32_t vs2,
                      uint32_t data, gen_helper_amo *fn, DisasContext *s)
{
    TCGv_ptr dest, mask, index;
    TCGv base;
    TCGv_i32 desc;

    TCGLabel *over = gen_new_label();
    tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    index = tcg_temp_new_ptr();
    base = get_gpr(s, rs1, EXT_NONE);
    desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));

    tcg_gen_addi_ptr(dest, cpu_env, vreg_ofs(s, vd));
    tcg_gen_addi_ptr(index, cpu_env, vreg_ofs(s, vs2));
    tcg_gen_addi_ptr(mask, cpu_env, vreg_ofs(s, 0));

    fn(dest, mask, base, index, cpu_env, desc);

    gen_set_label(over);
    return true;
}

static bool amo_op_7(DisasContext *s, arg_rwdvm *a, uint8_t seq)
{
    uint32_t data = 0;
    gen_helper_amo *fn;
    static gen_helper_amo *const fnsw[9] = {
        /* no atomic operation */
        gen_helper_vamoswapw_v_w,
        gen_helper_vamoaddw_v_w,
        gen_helper_vamoxorw_v_w,
        gen_helper_vamoandw_v_w,
        gen_helper_vamoorw_v_w,
        gen_helper_vamominw_v_w,
        gen_helper_vamomaxw_v_w,
        gen_helper_vamominuw_v_w,
        gen_helper_vamomaxuw_v_w
    };
    static gen_helper_amo *const fnsd[18] = {
        gen_helper_vamoswapw_v_d,
        gen_helper_vamoaddw_v_d,
        gen_helper_vamoxorw_v_d,
        gen_helper_vamoandw_v_d,
        gen_helper_vamoorw_v_d,
        gen_helper_vamominw_v_d,
        gen_helper_vamomaxw_v_d,
        gen_helper_vamominuw_v_d,
        gen_helper_vamomaxuw_v_d,
        gen_helper_vamoswapd_v_d,
        gen_helper_vamoaddd_v_d,
        gen_helper_vamoxord_v_d,
        gen_helper_vamoandd_v_d,
        gen_helper_vamoord_v_d,
        gen_helper_vamomind_v_d,
        gen_helper_vamomaxd_v_d,
        gen_helper_vamominud_v_d,
        gen_helper_vamomaxud_v_d
    };

    if (tb_cflags(s->base.tb) & CF_PARALLEL) {
        gen_helper_exit_atomic(cpu_env);
        s->base.is_jmp = DISAS_NORETURN;
        return true;
    }
    switch (s->sew) {
    case 0 ... 2:
        assert(seq < ARRAY_SIZE(fnsw));
        fn = fnsw[seq];
        break;
    case 3:
        /* XLEN check done in amo_check(). */
        assert(seq < ARRAY_SIZE(fnsd));
        fn = fnsd[seq];
        break;
    default:
        g_assert_not_reached();
    }

    data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_7, VM, a->vm);
    data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
    data = FIELD_DP32(data, VDATA_7, WD, a->wd);
    data = FIELD_DP32(data, VDATA_7, OL, get_xl(s));
    return amo_trans_7(a->rd, a->rs1, a->rs2, data, fn, s);
}
/*
 * There are two rules check here.
 *
 * 1. SEW must be at least as wide as the AMO memory element size.
 *
 * 2. If SEW is greater than XLEN, an illegal instruction exception is raised.
 */
static bool amo_check_7(DisasContext *s, arg_rwdvm* a)
{
    return (!s->vill && has_ext(s, RVA) &&
            (!a->wd || vext_check_overlap_mask(s, a->rd, a->vm, false)) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            ((1 << s->sew) <= (get_xlen(s) / 8)) &&
            ((1 << s->sew) >= 4));
}

static bool amo_check64_7(DisasContext *s, arg_rwdvm* a)
{
    REQUIRE_64BIT(s);
    return amo_check_7(s, a);
}

GEN_VEXT_TRANS(vamoswapw_v, 0, rwdvm, amo_op_7, amo_check_7)
GEN_VEXT_TRANS(vamoaddw_v, 1, rwdvm, amo_op_7, amo_check_7)
GEN_VEXT_TRANS(vamoxorw_v, 2, rwdvm, amo_op_7, amo_check_7)
GEN_VEXT_TRANS(vamoandw_v, 3, rwdvm, amo_op_7, amo_check_7)
GEN_VEXT_TRANS(vamoorw_v, 4, rwdvm, amo_op_7, amo_check_7)
GEN_VEXT_TRANS(vamominw_v, 5, rwdvm, amo_op_7, amo_check_7)
GEN_VEXT_TRANS(vamomaxw_v, 6, rwdvm, amo_op_7, amo_check_7)
GEN_VEXT_TRANS(vamominuw_v, 7, rwdvm, amo_op_7, amo_check_7)
GEN_VEXT_TRANS(vamomaxuw_v, 8, rwdvm, amo_op_7, amo_check_7)
GEN_VEXT_TRANS(vamoswapd_v, 9, rwdvm, amo_op_7, amo_check64_7)
GEN_VEXT_TRANS(vamoaddd_v, 10, rwdvm, amo_op_7, amo_check64_7)
GEN_VEXT_TRANS(vamoxord_v, 11, rwdvm, amo_op_7, amo_check64_7)
GEN_VEXT_TRANS(vamoandd_v, 12, rwdvm, amo_op_7, amo_check64_7)
GEN_VEXT_TRANS(vamoord_v, 13, rwdvm, amo_op_7, amo_check64_7)
GEN_VEXT_TRANS(vamomind_v, 14, rwdvm, amo_op_7, amo_check64_7)
GEN_VEXT_TRANS(vamomaxd_v, 15, rwdvm, amo_op_7, amo_check64_7)
GEN_VEXT_TRANS(vamominud_v, 16, rwdvm, amo_op_7, amo_check64_7)
GEN_VEXT_TRANS(vamomaxud_v, 17, rwdvm, amo_op_7, amo_check64_7)

/*
 *** Vector Integer Arithmetic Instructions
 */
#define MAXSZ_7(s) (s->vlen >> (3 - s->lmul))

static bool opivv_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_reg(s, a->rs1, false));
}

static inline bool
do_opivv_gvec_7(DisasContext *s, arg_rmrr *a, GVecGen3Fn *gvec_fn,
              gen_helper_gvec_4_ptr *fn)
{
    TCGLabel *over = gen_new_label();
    if (!opivv_check_7(s, a)) {
        return false;
    }

    tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

    if (a->vm && s->vl_eq_vlmax) {
        gvec_fn(s->sew, vreg_ofs(s, a->rd),
                vreg_ofs(s, a->rs2), vreg_ofs(s, a->rs1),
                MAXSZ_7(s), MAXSZ_7(s));
    } else {
        uint32_t data = 0;

        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
        tcg_gen_gvec_4_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),
                           vreg_ofs(s, a->rs1), vreg_ofs(s, a->rs2),
                           cpu_env, s->vlen / 8, s->vlen / 8, data, fn);
    }
    gen_set_label(over);
    return true;
}

/* OPIVV with GVEC IR */
#undef GEN_OPIVV_GVEC_TRANS
#define GEN_OPIVV_GVEC_TRANS(NAME, SUF) \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)           \
{                                                                  \
    static gen_helper_gvec_4_ptr * const fns[4] = {                \
        gen_helper_##NAME##_b, gen_helper_##NAME##_h,              \
        gen_helper_##NAME##_w, gen_helper_##NAME##_d,              \
    };                                                             \
    return do_opivv_gvec_7(s, a, tcg_gen_gvec_##SUF, fns[s->sew]);   \
}

GEN_OPIVV_GVEC_TRANS(vadd_vv_7, add)
GEN_OPIVV_GVEC_TRANS(vsub_vv_7, sub)

static bool opivx_trans_7(uint32_t vd, uint32_t rs1, uint32_t vs2, uint32_t vm,
                        gen_helper_opivx *fn, DisasContext *s)
{
    TCGv_ptr dest, src2, mask;
    TCGv src1;
    TCGv_i32 desc;
    uint32_t data = 0;

    TCGLabel *over = gen_new_label();
    tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    src2 = tcg_temp_new_ptr();
    src1 = get_gpr(s, rs1, EXT_SIGN);

    data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_7, VM, vm);
    data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
    desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));

    tcg_gen_addi_ptr(dest, cpu_env, vreg_ofs(s, vd));
    tcg_gen_addi_ptr(src2, cpu_env, vreg_ofs(s, vs2));
    tcg_gen_addi_ptr(mask, cpu_env, vreg_ofs(s, 0));

    fn(dest, mask, src1, src2, cpu_env, desc);

    gen_set_label(over);
    return true;
}

static bool opivx_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false));
}

static inline bool
do_opivx_gvec_7(DisasContext *s, arg_rmrr *a, GVecGen2sFn *gvec_fn,
                gen_helper_opivx *fn)
{
    if (!opivx_check_7(s, a)) {
        return false;
    }

    if (a->vm && s->vl_eq_vlmax) {
        TCGv_i64 src1 = tcg_temp_new_i64();

        tcg_gen_ext_tl_i64(src1, get_gpr(s, a->rs1, EXT_SIGN));
        gvec_fn(s->sew, vreg_ofs(s, a->rd), vreg_ofs(s, a->rs2),
                src1, MAXSZ_7(s), MAXSZ_7(s));

        return true;
    }
    return opivx_trans_7(a->rd, a->rs1, a->rs2, a->vm, fn, s);
}

/* OPIVX with GVEC IR */
#undef GEN_OPIVX_GVEC_TRANS
#define GEN_OPIVX_GVEC_TRANS(NAME, SUF) \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    static gen_helper_opivx * const fns[4] = {                     \
        gen_helper_##NAME##_b, gen_helper_##NAME##_h,              \
        gen_helper_##NAME##_w, gen_helper_##NAME##_d,              \
    };                                                             \
    return do_opivx_gvec_7(s, a, tcg_gen_gvec_##SUF, fns[s->sew]); \
}

GEN_OPIVX_GVEC_TRANS(vadd_vx_7, adds)
GEN_OPIVX_GVEC_TRANS(vsub_vx_7, subs)
GEN_OPIVX_GVEC_TRANS(vrsub_vx_7, rsubs)

typedef enum {
    IMM_ZX_7,         /* Zero-extended */
    IMM_SX_7,         /* Sign-extended */
    IMM_TRUNC_SEW_7,  /* Truncate to log(SEW) bits */
    IMM_TRUNC_2SEW_7, /* Truncate to log(2*SEW) bits */
} imm_mode_t_7;

static int64_t extract_imm_7(DisasContext *s, uint32_t imm, imm_mode_t_7 imm_mode)
{
    switch (imm_mode) {
    case IMM_ZX_7:
        return extract64(imm, 0, 5);
    case IMM_SX_7:
        return sextract64(imm, 0, 5);
    case IMM_TRUNC_SEW_7:
        return extract64(imm, 0, s->sew + 3);
    case IMM_TRUNC_2SEW_7:
        return extract64(imm, 0, s->sew + 4);
    default:
        g_assert_not_reached();
    }
}

static bool opivi_trans_7(uint32_t vd, uint32_t imm, uint32_t vs2, uint32_t vm,
                        gen_helper_opivx *fn, DisasContext *s,
                        imm_mode_t_7 imm_mode)
{
    TCGv_ptr dest, src2, mask;
    TCGv src1;
    TCGv_i32 desc;
    uint32_t data = 0;

    TCGLabel *over = gen_new_label();
    tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    src2 = tcg_temp_new_ptr();
    src1 = tcg_constant_tl(extract_imm_7(s, imm, imm_mode));

    data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
    data = FIELD_DP32(data, VDATA_7, VM, vm);
    data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
    desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));

    tcg_gen_addi_ptr(dest, cpu_env, vreg_ofs(s, vd));
    tcg_gen_addi_ptr(src2, cpu_env, vreg_ofs(s, vs2));
    tcg_gen_addi_ptr(mask, cpu_env, vreg_ofs(s, 0));

    fn(dest, mask, src1, src2, cpu_env, desc);

    gen_set_label(over);
    return true;
}

static inline bool
do_opivi_gvec_7(DisasContext *s, arg_rmrr *a, GVecGen2iFn *gvec_fn,
              gen_helper_opivx *fn, imm_mode_t_7 imm_mode)
{
    if (!opivx_check_7(s, a)) {
        return false;
    }

    if (a->vm && s->vl_eq_vlmax) {
        gvec_fn(s->sew, vreg_ofs(s, a->rd), vreg_ofs(s, a->rs2),
                extract_imm_7(s, a->rs1, imm_mode), MAXSZ_7(s), MAXSZ_7(s));
    } else {
        return opivi_trans_7(a->rd, a->rs1, a->rs2, a->vm, fn, s, imm_mode);
    }
    return true;
}

/* OPIVI with GVEC IR */
#undef GEN_OPIVI_GVEC_TRANS
#define GEN_OPIVI_GVEC_TRANS(NAME, IMM_MODE, OPIVX, SUF) \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    static gen_helper_opivx * const fns[4] = {                     \
        gen_helper_##OPIVX##_b, gen_helper_##OPIVX##_h,            \
        gen_helper_##OPIVX##_w, gen_helper_##OPIVX##_d,            \
    };                                                             \
    return do_opivi_gvec_7(s, a, tcg_gen_gvec_##SUF,               \
                         fns[s->sew], IMM_MODE);                   \
}

GEN_OPIVI_GVEC_TRANS(vadd_vi_7, IMM_SX_7, vadd_vx_7, addi)
GEN_OPIVI_GVEC_TRANS(vrsub_vi_7, IMM_SX_7, vrsub_vx_7, rsubi)

/* Vector Widening Integer Add/Subtract */

/* OPIVV with WIDEN */
static bool opivv_widen_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, true) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_reg(s, a->rs1, false) &&
            vext_check_overlap_group(a->rd, 2 << s->lmul, a->rs2,
                                     1 << s->lmul) &&
            vext_check_overlap_group(a->rd, 2 << s->lmul, a->rs1,
                                     1 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

static bool do_opivv_widen_7(DisasContext *s, arg_rmrr *a,
                           gen_helper_gvec_4_ptr *fn,
                           bool (*checkfn)(DisasContext *, arg_rmrr *))
{
    if (checkfn(s, a)) {
        uint32_t data = 0;
        TCGLabel *over = gen_new_label();
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
        tcg_gen_gvec_4_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),
                           vreg_ofs(s, a->rs1),
                           vreg_ofs(s, a->rs2),
                           cpu_env, s->vlen / 8, s->vlen / 8,
                           data, fn);
        gen_set_label(over);
        return true;
    }
    return false;
}

#undef GEN_OPIVV_WIDEN_TRANS
#define GEN_OPIVV_WIDEN_TRANS(NAME, CHECK) \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)       \
{                                                            \
    static gen_helper_gvec_4_ptr * const fns[3] = {          \
        gen_helper_##NAME##_b,                               \
        gen_helper_##NAME##_h,                               \
        gen_helper_##NAME##_w                                \
    };                                                       \
    return do_opivv_widen_7(s, a, fns[s->sew], CHECK);         \
}

GEN_OPIVV_WIDEN_TRANS(vwaddu_vv_7, opivv_widen_check_7)
GEN_OPIVV_WIDEN_TRANS(vwadd_vv_7, opivv_widen_check_7)
GEN_OPIVV_WIDEN_TRANS(vwsubu_vv_7, opivv_widen_check_7)
GEN_OPIVV_WIDEN_TRANS(vwsub_vv_7, opivv_widen_check_7)

/* OPIVX with WIDEN */
static bool opivx_widen_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, true) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_overlap_group(a->rd, 2 << s->lmul, a->rs2,
                                     1 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

static bool do_opivx_widen_7(DisasContext *s, arg_rmrr *a,
                           gen_helper_opivx *fn)
{
    if (opivx_widen_check_7(s, a)) {
        return opivx_trans_7(a->rd, a->rs1, a->rs2, a->vm, fn, s);
    }
    return false;
}

#undef GEN_OPIVX_WIDEN_TRANS
#define GEN_OPIVX_WIDEN_TRANS(NAME) \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)       \
{                                                            \
    static gen_helper_opivx * const fns[3] = {               \
        gen_helper_##NAME##_b,                               \
        gen_helper_##NAME##_h,                               \
        gen_helper_##NAME##_w                                \
    };                                                       \
    return do_opivx_widen_7(s, a, fns[s->sew]);                \
}

GEN_OPIVX_WIDEN_TRANS(vwaddu_vx_7)
GEN_OPIVX_WIDEN_TRANS(vwadd_vx_7)
GEN_OPIVX_WIDEN_TRANS(vwsubu_vx_7)
GEN_OPIVX_WIDEN_TRANS(vwsub_vx_7)

/* WIDEN OPIVV with WIDEN */
static bool opiwv_widen_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, true) &&
            vext_check_reg(s, a->rs2, true) &&
            vext_check_reg(s, a->rs1, false) &&
            vext_check_overlap_group(a->rd, 2 << s->lmul, a->rs1,
                                     1 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

static bool do_opiwv_widen_7(DisasContext *s, arg_rmrr *a,
                           gen_helper_gvec_4_ptr *fn)
{
    if (opiwv_widen_check_7(s, a)) {
        uint32_t data = 0;
        TCGLabel *over = gen_new_label();
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
        tcg_gen_gvec_4_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),
                           vreg_ofs(s, a->rs1),
                           vreg_ofs(s, a->rs2),
                           cpu_env, s->vlen / 8, s->vlen / 8, data, fn);
        gen_set_label(over);
        return true;
    }
    return false;
}

#undef GEN_OPIWV_WIDEN_TRANS
#define GEN_OPIWV_WIDEN_TRANS(NAME) \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)       \
{                                                            \
    static gen_helper_gvec_4_ptr * const fns[3] = {          \
        gen_helper_##NAME##_b,                               \
        gen_helper_##NAME##_h,                               \
        gen_helper_##NAME##_w                                \
    };                                                       \
    return do_opiwv_widen_7(s, a, fns[s->sew]);                \
}

GEN_OPIWV_WIDEN_TRANS(vwaddu_wv_7)
GEN_OPIWV_WIDEN_TRANS(vwadd_wv_7)
GEN_OPIWV_WIDEN_TRANS(vwsubu_wv_7)
GEN_OPIWV_WIDEN_TRANS(vwsub_wv_7)

/* WIDEN OPIVX with WIDEN */
static bool opiwx_widen_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, true) &&
            vext_check_reg(s, a->rs2, true) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

static bool do_opiwx_widen_7(DisasContext *s, arg_rmrr *a,
                           gen_helper_opivx *fn)
{
    if (opiwx_widen_check_7(s, a)) {
        return opivx_trans_7(a->rd, a->rs1, a->rs2, a->vm, fn, s);
    }
    return false;
}

#undef GEN_OPIWX_WIDEN_TRANS
#define GEN_OPIWX_WIDEN_TRANS(NAME) \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)       \
{                                                            \
    static gen_helper_opivx * const fns[3] = {               \
        gen_helper_##NAME##_b,                               \
        gen_helper_##NAME##_h,                               \
        gen_helper_##NAME##_w                                \
    };                                                       \
    return do_opiwx_widen_7(s, a, fns[s->sew]);                \
}

GEN_OPIWX_WIDEN_TRANS(vwaddu_wx_7)
GEN_OPIWX_WIDEN_TRANS(vwadd_wx_7)
GEN_OPIWX_WIDEN_TRANS(vwsubu_wx_7)
GEN_OPIWX_WIDEN_TRANS(vwsub_wx_7)

/* Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions */
/* OPIVV without GVEC IR */
#undef GEN_OPIVV_TRANS
#define GEN_OPIVV_TRANS(NAME, CHECK)                               \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    if (CHECK(s, a)) {                                             \
        uint32_t data = 0;                                         \
        static gen_helper_gvec_4_ptr * const fns[4] = {            \
            gen_helper_##NAME##_b, gen_helper_##NAME##_h,          \
            gen_helper_##NAME##_w, gen_helper_##NAME##_d,          \
        };                                                         \
        TCGLabel *over = gen_new_label();                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);             \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);                 \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);             \
        tcg_gen_gvec_4_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),     \
                           vreg_ofs(s, a->rs1),                    \
                           vreg_ofs(s, a->rs2), cpu_env, s->vlen / 8, \
                           s->vlen / 8, data, fns[s->sew]);        \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}

/*
 * For vadc and vsbc, an illegal instruction exception is raised if the
 * destination vector register is v0 and LMUL > 1. (Section 12.3)
 */
static bool opivv_vadc_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_reg(s, a->rs1, false) &&
            ((a->rd != 0) || (s->lmul == 0)));
}

GEN_OPIVV_TRANS(vadc_vvm_7, opivv_vadc_check_7)
GEN_OPIVV_TRANS(vsbc_vvm_7, opivv_vadc_check_7)

/*
 * For vmadc and vmsbc, an illegal instruction exception is raised if the
 * destination vector register overlaps a source vector register group.
 */
static bool opivv_vmadc_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_reg(s, a->rs1, false) &&
            vext_check_overlap_group(a->rd, 1, a->rs1, 1 << s->lmul) &&
            vext_check_overlap_group(a->rd, 1, a->rs2, 1 << s->lmul));
}

GEN_OPIVV_TRANS(vmadc_vvm_7, opivv_vmadc_check_7)
GEN_OPIVV_TRANS(vmsbc_vvm_7, opivv_vmadc_check_7)

static bool opivx_vadc_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            ((a->rd != 0) || (s->lmul == 0)));
}

/* OPIVX without GVEC IR */
#undef GEN_OPIVX_TRANS
#define GEN_OPIVX_TRANS(NAME, CHECK)                                     \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)                   \
{                                                                        \
    if (CHECK(s, a)) {                                                   \
        static gen_helper_opivx * const fns[4] = {                       \
            gen_helper_##NAME##_b, gen_helper_##NAME##_h,                \
            gen_helper_##NAME##_w, gen_helper_##NAME##_d,                \
        };                                                               \
                                                                         \
        return opivx_trans_7(a->rd, a->rs1, a->rs2, a->vm,               \
                             fns[s->sew], s);                            \
    }                                                                    \
    return false;                                                        \
}

GEN_OPIVX_TRANS(vadc_vxm_7, opivx_vadc_check_7)
GEN_OPIVX_TRANS(vsbc_vxm_7, opivx_vadc_check_7)

static bool opivx_vmadc_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_overlap_group(a->rd, 1, a->rs2, 1 << s->lmul));
}

GEN_OPIVX_TRANS(vmadc_vxm_7, opivx_vmadc_check_7)
GEN_OPIVX_TRANS(vmsbc_vxm_7, opivx_vmadc_check_7)

/* OPIVI without GVEC IR */
#undef GEN_OPIVI_TRANS
#define GEN_OPIVI_TRANS(NAME, ZX, OPIVX, CHECK)                          \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)                   \
{                                                                        \
    if (CHECK(s, a)) {                                                   \
        static gen_helper_opivx * const fns[4] = {                       \
            gen_helper_##OPIVX##_b, gen_helper_##OPIVX##_h,              \
            gen_helper_##OPIVX##_w, gen_helper_##OPIVX##_d,              \
        };                                                               \
        return opivi_trans_7(a->rd, a->rs1, a->rs2, a->vm,                 \
                           fns[s->sew], s, ZX);                          \
    }                                                                    \
    return false;                                                        \
}

GEN_OPIVI_TRANS(vadc_vim_7, IMM_SX_7, vadc_vxm_7, opivx_vadc_check_7)
GEN_OPIVI_TRANS(vmadc_vim_7, IMM_SX_7, vmadc_vxm_7, opivx_vmadc_check_7)

/* Vector Bitwise Logical Instructions */
GEN_OPIVV_GVEC_TRANS(vand_vv_7, and)
GEN_OPIVV_GVEC_TRANS(vor_vv_7,  or)
GEN_OPIVV_GVEC_TRANS(vxor_vv_7, xor)
GEN_OPIVX_GVEC_TRANS(vand_vx_7, ands)
GEN_OPIVX_GVEC_TRANS(vor_vx_7,  ors)
GEN_OPIVX_GVEC_TRANS(vxor_vx_7, xors)
GEN_OPIVI_GVEC_TRANS(vand_vi_7, IMM_SX_7, vand_vx_7, andi)
GEN_OPIVI_GVEC_TRANS(vor_vi_7, IMM_SX_7, vor_vx_7,  ori)
GEN_OPIVI_GVEC_TRANS(vxor_vi_7, IMM_SX_7, vxor_vx_7, xori)

/* Vector Single-Width Bit Shift Instructions */
GEN_OPIVV_GVEC_TRANS(vsll_vv_7,  shlv)
GEN_OPIVV_GVEC_TRANS(vsrl_vv_7,  shrv)
GEN_OPIVV_GVEC_TRANS(vsra_vv_7,  sarv)

static inline bool
do_opivx_gvec_shift_7(DisasContext *s, arg_rmrr *a, GVecGen2sFn32 *gvec_fn,
                    gen_helper_opivx *fn)
{
    if (!opivx_check_7(s, a)) {
        return false;
    }

    if (a->vm && s->vl_eq_vlmax) {
        TCGv_i32 src1 = tcg_temp_new_i32();

        tcg_gen_trunc_tl_i32(src1, get_gpr(s, a->rs1, EXT_NONE));
        tcg_gen_extract_i32(src1, src1, 0, s->sew + 3);
        gvec_fn(s->sew, vreg_ofs(s, a->rd), vreg_ofs(s, a->rs2),
                src1, MAXSZ_7(s), MAXSZ_7(s));

        return true;
    }
    return opivx_trans_7(a->rd, a->rs1, a->rs2, a->vm, fn, s);
}

#undef GEN_OPIVX_GVEC_SHIFT_TRANS
#define GEN_OPIVX_GVEC_SHIFT_TRANS(NAME, SUF) \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)                    \
{                                                                         \
    static gen_helper_opivx * const fns[4] = {                            \
        gen_helper_##NAME##_b, gen_helper_##NAME##_h,                     \
        gen_helper_##NAME##_w, gen_helper_##NAME##_d,                     \
    };                                                                    \
                                                                          \
    return do_opivx_gvec_shift_7(s, a, tcg_gen_gvec_##SUF, fns[s->sew]);    \
}

GEN_OPIVX_GVEC_SHIFT_TRANS(vsll_vx_7,  shls)
GEN_OPIVX_GVEC_SHIFT_TRANS(vsrl_vx_7,  shrs)
GEN_OPIVX_GVEC_SHIFT_TRANS(vsra_vx_7,  sars)

GEN_OPIVI_GVEC_TRANS(vsll_vi_7, IMM_TRUNC_SEW_7, vsll_vx_7,  shli)
GEN_OPIVI_GVEC_TRANS(vsrl_vi_7, IMM_TRUNC_SEW_7, vsrl_vx_7,  shri)
GEN_OPIVI_GVEC_TRANS(vsra_vi_7, IMM_TRUNC_SEW_7, vsra_vx_7,  sari)

/* Vector Narrowing Integer Right Shift Instructions */
static bool opivv_narrow_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, true) &&
            vext_check_reg(s, a->rs1, false) &&
            vext_check_overlap_group(a->rd, 1 << s->lmul, a->rs2,
                2 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

/* OPIVV with NARROW */
#undef GEN_OPIVV_NARROW_TRANS
#define GEN_OPIVV_NARROW_TRANS(NAME)                               \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    if (opivv_narrow_check_7(s, a)) {                                \
        uint32_t data = 0;                                         \
        static gen_helper_gvec_4_ptr * const fns[3] = {            \
            gen_helper_##NAME##_b,                                 \
            gen_helper_##NAME##_h,                                 \
            gen_helper_##NAME##_w,                                 \
        };                                                         \
        TCGLabel *over = gen_new_label();                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);             \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);                 \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);             \
        tcg_gen_gvec_4_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),     \
                           vreg_ofs(s, a->rs1),                    \
                           vreg_ofs(s, a->rs2), cpu_env, s->vlen / 8, \
                           s->vlen / 8, data, fns[s->sew]);        \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}
GEN_OPIVV_NARROW_TRANS(vnsra_vv_7)
GEN_OPIVV_NARROW_TRANS(vnsrl_vv_7)

static bool opivx_narrow_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, true) &&
            vext_check_overlap_group(a->rd, 1 << s->lmul, a->rs2,
                2 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

/* OPIVX with NARROW */
#undef GEN_OPIVX_NARROW_TRANS
#define GEN_OPIVX_NARROW_TRANS(NAME)                                     \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)                   \
{                                                                        \
    if (opivx_narrow_check_7(s, a)) {                                      \
        static gen_helper_opivx * const fns[3] = {                       \
            gen_helper_##NAME##_b,                                       \
            gen_helper_##NAME##_h,                                       \
            gen_helper_##NAME##_w,                                       \
        };                                                               \
        return opivx_trans_7(a->rd, a->rs1, a->rs2, a->vm, fns[s->sew], s);\
    }                                                                    \
    return false;                                                        \
}

GEN_OPIVX_NARROW_TRANS(vnsra_vx_7)
GEN_OPIVX_NARROW_TRANS(vnsrl_vx_7)

/* OPIVI with NARROW */
#undef GEN_OPIVI_NARROW_TRANS
#define GEN_OPIVI_NARROW_TRANS(NAME, ZX, OPIVX)                          \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)                   \
{                                                                        \
    if (opivx_narrow_check_7(s, a)) {                                      \
        static gen_helper_opivx * const fns[3] = {                       \
            gen_helper_##OPIVX##_b,                                      \
            gen_helper_##OPIVX##_h,                                      \
            gen_helper_##OPIVX##_w,                                      \
        };                                                               \
        return opivi_trans_7(a->rd, a->rs1, a->rs2, a->vm,                 \
                           fns[s->sew], s, ZX);                          \
    }                                                                    \
    return false;                                                        \
}

GEN_OPIVI_NARROW_TRANS(vnsra_vi_7, IMM_ZX, vnsra_vx_7)
GEN_OPIVI_NARROW_TRANS(vnsrl_vi_7, IMM_ZX, vnsrl_vx_7)

/* Vector Integer Comparison Instructions */
/*
 * For all comparison instructions, an illegal instruction exception is raised
 * if the destination vector register overlaps a source vector register group
 * and LMUL > 1.
 */
static bool opivv_cmp_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_reg(s, a->rs1, false) &&
            ((vext_check_overlap_group(a->rd, 1, a->rs1, 1 << s->lmul) &&
              vext_check_overlap_group(a->rd, 1, a->rs2, 1 << s->lmul)) ||
             (s->lmul == 0)));
}
GEN_OPIVV_TRANS(vmseq_vv_7, opivv_cmp_check_7)
GEN_OPIVV_TRANS(vmsne_vv_7, opivv_cmp_check_7)
GEN_OPIVV_TRANS(vmsltu_vv_7, opivv_cmp_check_7)
GEN_OPIVV_TRANS(vmslt_vv_7, opivv_cmp_check_7)
GEN_OPIVV_TRANS(vmsleu_vv_7, opivv_cmp_check_7)
GEN_OPIVV_TRANS(vmsle_vv_7, opivv_cmp_check_7)

static bool opivx_cmp_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rs2, false) &&
            (vext_check_overlap_group(a->rd, 1, a->rs2, 1 << s->lmul) ||
             (s->lmul == 0)));
}

GEN_OPIVX_TRANS(vmseq_vx_7, opivx_cmp_check_7)
GEN_OPIVX_TRANS(vmsne_vx_7, opivx_cmp_check_7)
GEN_OPIVX_TRANS(vmsltu_vx_7, opivx_cmp_check_7)
GEN_OPIVX_TRANS(vmslt_vx_7, opivx_cmp_check_7)
GEN_OPIVX_TRANS(vmsleu_vx_7, opivx_cmp_check_7)
GEN_OPIVX_TRANS(vmsle_vx_7, opivx_cmp_check_7)
GEN_OPIVX_TRANS(vmsgtu_vx_7, opivx_cmp_check_7)
GEN_OPIVX_TRANS(vmsgt_vx_7, opivx_cmp_check_7)

GEN_OPIVI_TRANS(vmseq_vi_7, IMM_SX_7, vmseq_vx_7, opivx_cmp_check_7)
GEN_OPIVI_TRANS(vmsne_vi_7, IMM_SX_7, vmsne_vx_7, opivx_cmp_check_7)
GEN_OPIVI_TRANS(vmsleu_vi_7, IMM_ZX_7, vmsleu_vx_7, opivx_cmp_check_7)
GEN_OPIVI_TRANS(vmsle_vi_7, IMM_SX_7, vmsle_vx_7, opivx_cmp_check_7)
GEN_OPIVI_TRANS(vmsgtu_vi_7, IMM_ZX_7, vmsgtu_vx_7, opivx_cmp_check_7)
GEN_OPIVI_TRANS(vmsgt_vi_7, IMM_SX_7, vmsgt_vx_7, opivx_cmp_check_7)

/* Vector Integer Min/Max Instructions */
GEN_OPIVV_GVEC_TRANS(vminu_vv_7, umin)
GEN_OPIVV_GVEC_TRANS(vmin_vv_7,  smin)
GEN_OPIVV_GVEC_TRANS(vmaxu_vv_7, umax)
GEN_OPIVV_GVEC_TRANS(vmax_vv_7,  smax)
GEN_OPIVX_TRANS(vminu_vx_7, opivx_check_7)
GEN_OPIVX_TRANS(vmin_vx_7,  opivx_check_7)
GEN_OPIVX_TRANS(vmaxu_vx_7, opivx_check_7)
GEN_OPIVX_TRANS(vmax_vx_7,  opivx_check_7)

/* Vector Single-Width Integer Multiply Instructions */
GEN_OPIVV_GVEC_TRANS(vmul_vv_7,  mul)
GEN_OPIVV_TRANS(vmulh_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vmulhu_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vmulhsu_vv_7, opivv_check_7)
GEN_OPIVX_GVEC_TRANS(vmul_vx_7,  muls)
GEN_OPIVX_TRANS(vmulh_vx_7, opivx_check_7)
GEN_OPIVX_TRANS(vmulhu_vx_7, opivx_check_7)
GEN_OPIVX_TRANS(vmulhsu_vx_7, opivx_check_7)

/* Vector Integer Divide Instructions */
GEN_OPIVV_TRANS(vdivu_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vdiv_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vremu_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vrem_vv_7, opivv_check_7)
GEN_OPIVX_TRANS(vdivu_vx_7, opivx_check_7)
GEN_OPIVX_TRANS(vdiv_vx_7, opivx_check_7)
GEN_OPIVX_TRANS(vremu_vx_7, opivx_check_7)
GEN_OPIVX_TRANS(vrem_vx_7, opivx_check_7)

/* Vector Widening Integer Multiply Instructions */
GEN_OPIVV_WIDEN_TRANS(vwmul_vv_7, opivv_widen_check_7)
GEN_OPIVV_WIDEN_TRANS(vwmulu_vv_7, opivv_widen_check_7)
GEN_OPIVV_WIDEN_TRANS(vwmulsu_vv_7, opivv_widen_check_7)
GEN_OPIVX_WIDEN_TRANS(vwmul_vx_7)
GEN_OPIVX_WIDEN_TRANS(vwmulu_vx_7)
GEN_OPIVX_WIDEN_TRANS(vwmulsu_vx_7)

/* Vector Single-Width Integer Multiply-Add Instructions */
GEN_OPIVV_TRANS(vmacc_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vnmsac_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vmadd_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vnmsub_vv_7, opivv_check_7)
GEN_OPIVX_TRANS(vmacc_vx_7, opivx_check_7)
GEN_OPIVX_TRANS(vnmsac_vx_7, opivx_check_7)
GEN_OPIVX_TRANS(vmadd_vx_7, opivx_check_7)
GEN_OPIVX_TRANS(vnmsub_vx_7, opivx_check_7)

/* Vector Widening Integer Multiply-Add Instructions */
GEN_OPIVV_WIDEN_TRANS(vwmaccu_vv_7, opivv_widen_check_7)
GEN_OPIVV_WIDEN_TRANS(vwmacc_vv_7, opivv_widen_check_7)
GEN_OPIVV_WIDEN_TRANS(vwmaccsu_vv_7, opivv_widen_check_7)
GEN_OPIVX_WIDEN_TRANS(vwmaccu_vx_7)
GEN_OPIVX_WIDEN_TRANS(vwmacc_vx_7)
GEN_OPIVX_WIDEN_TRANS(vwmaccsu_vx_7)
GEN_OPIVX_WIDEN_TRANS(vwmaccus_vx_7)

/* Vector Integer Merge and Move Instructions */
static bool trans_vmv_v_v_7(DisasContext *s, arg_vmv_v_v_7 *a)
{
    if (vext_check_isa_ill(s) &&
        vext_check_reg(s, a->rd, false) &&
        vext_check_reg(s, a->rs1, false)) {

        if (s->vl_eq_vlmax) {
            tcg_gen_gvec_mov(s->sew, vreg_ofs(s, a->rd),
                             vreg_ofs(s, a->rs1),
                             MAXSZ_7(s), MAXSZ_7(s));
        } else {
            uint32_t data = FIELD_DP32(0, VDATA_7, LMUL, s->lmul);
            static gen_helper_gvec_2_ptr * const fns[4] = {
                gen_helper_vmv_v_v_7_b, gen_helper_vmv_v_v_7_h,
                gen_helper_vmv_v_v_7_w, gen_helper_vmv_v_v_7_d,
            };
            TCGLabel *over = gen_new_label();
            tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

            tcg_gen_gvec_2_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, a->rs1),
                               cpu_env, s->vlen / 8, s->vlen / 8, data, fns[s->sew]);
            gen_set_label(over);
        }
        return true;
    }
    return false;
}

static bool trans_vmv_v_x_7(DisasContext *s, arg_vmv_v_x_7 *a)
{
    if (vext_check_isa_ill(s) &&
        vext_check_reg(s, a->rd, false)) {

        TCGv s1;
        TCGLabel *over = gen_new_label();
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

        s1 = get_gpr(s, a->rs1, EXT_SIGN);

        if (s->vl_eq_vlmax && (8 << s->sew) <= get_xlen(s)) {
            tcg_gen_gvec_dup_tl(s->sew, vreg_ofs(s, a->rd),
                                MAXSZ_7(s), MAXSZ_7(s), s1);
        } else {
            TCGv_i32 desc;
            TCGv_i64 s1_i64 = tcg_temp_new_i64();
            TCGv_ptr dest = tcg_temp_new_ptr();
            uint32_t data = FIELD_DP32(0, VDATA_7, LMUL, s->lmul);
            static gen_helper_vmv_vx * const fns[4] = {
                gen_helper_vmv_v_x_7_b, gen_helper_vmv_v_x_7_h,
                gen_helper_vmv_v_x_7_w, gen_helper_vmv_v_x_7_d,
            };

            tcg_gen_ext_tl_i64(s1_i64, s1);
            desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));
            tcg_gen_addi_ptr(dest, cpu_env, vreg_ofs(s, a->rd));
            fns[s->sew](dest, s1_i64, cpu_env, desc);
        }

        gen_set_label(over);
        return true;
    }
    return false;
}

static bool trans_vmv_v_i_7(DisasContext *s, arg_vmv_v_i_7 *a)
{
    if (vext_check_isa_ill(s) &&
        vext_check_reg(s, a->rd, false)) {

        int64_t simm = sextract64(a->rs1, 0, 5);
        if (s->vl_eq_vlmax) {
            tcg_gen_gvec_dup_imm(s->sew, vreg_ofs(s, a->rd),
                                 MAXSZ_7(s), MAXSZ_7(s), simm);
        } else {
            TCGv_i32 desc;
            TCGv_i64 s1;
            TCGv_ptr dest;
            uint32_t data = FIELD_DP32(0, VDATA_7, LMUL, s->lmul);
            static gen_helper_vmv_vx * const fns[4] = {
                gen_helper_vmv_v_x_7_b, gen_helper_vmv_v_x_7_h,
                gen_helper_vmv_v_x_7_w, gen_helper_vmv_v_x_7_d,
            };
            TCGLabel *over = gen_new_label();
            tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

            s1 = tcg_constant_i64(simm);
            dest = tcg_temp_new_ptr();
            desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));
            tcg_gen_addi_ptr(dest, cpu_env, vreg_ofs(s, a->rd));
            fns[s->sew](dest, s1, cpu_env, desc);

            gen_set_label(over);
        }
        return true;
    }
    return false;
}

GEN_OPIVV_TRANS(vmerge_vvm_7, opivv_vadc_check_7)
GEN_OPIVX_TRANS(vmerge_vxm_7, opivx_vadc_check_7)
GEN_OPIVI_TRANS(vmerge_vim_7, IMM_SX_7, vmerge_vxm_7, opivx_vadc_check_7)

/*
 *** Vector Fixed-Point Arithmetic Instructions
 */

/* Vector Single-Width Saturating Add and Subtract */
GEN_OPIVV_TRANS(vsaddu_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vsadd_vv_7,  opivv_check_7)
GEN_OPIVV_TRANS(vssubu_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vssub_vv_7,  opivv_check_7)
GEN_OPIVX_TRANS(vsaddu_vx_7,  opivx_check_7)
GEN_OPIVX_TRANS(vsadd_vx_7,  opivx_check_7)
GEN_OPIVX_TRANS(vssubu_vx_7,  opivx_check_7)
GEN_OPIVX_TRANS(vssub_vx_7,  opivx_check_7)
GEN_OPIVI_TRANS(vsaddu_vi_7, IMM_ZX_7, vsaddu_vx_7, opivx_check_7)
GEN_OPIVI_TRANS(vsadd_vi_7, IMM_SX_7, vsadd_vx_7, opivx_check_7)

/* Vector Single-Width Averaging Add and Subtract */
GEN_OPIVV_TRANS(vaadd_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vasub_vv_7, opivv_check_7)
GEN_OPIVX_TRANS(vaadd_vx_7,  opivx_check_7)
GEN_OPIVX_TRANS(vasub_vx_7,  opivx_check_7)
GEN_OPIVI_TRANS(vaadd_vi_7, IMM_SX_7, vaadd_vx_7, opivx_check_7)

/* Vector Single-Width Fractional Multiply with Rounding and Saturation */
GEN_OPIVV_TRANS(vsmul_vv_7, opivv_check_7)
GEN_OPIVX_TRANS(vsmul_vx_7,  opivx_check_7)

/* Vector Widening Saturating Scaled Multiply-Add */
GEN_OPIVV_WIDEN_TRANS(vwsmaccu_vv_7, opivv_widen_check_7)
GEN_OPIVV_WIDEN_TRANS(vwsmacc_vv_7, opivv_widen_check_7)
GEN_OPIVV_WIDEN_TRANS(vwsmaccsu_vv_7, opivv_widen_check_7)
GEN_OPIVX_WIDEN_TRANS(vwsmaccu_vx_7)
GEN_OPIVX_WIDEN_TRANS(vwsmacc_vx_7)
GEN_OPIVX_WIDEN_TRANS(vwsmaccsu_vx_7)
GEN_OPIVX_WIDEN_TRANS(vwsmaccus_vx_7)

/* Vector Single-Width Scaling Shift Instructions */
GEN_OPIVV_TRANS(vssrl_vv_7, opivv_check_7)
GEN_OPIVV_TRANS(vssra_vv_7, opivv_check_7)
GEN_OPIVX_TRANS(vssrl_vx_7,  opivx_check_7)
GEN_OPIVX_TRANS(vssra_vx_7,  opivx_check_7)
GEN_OPIVI_TRANS(vssrl_vi_7, IMM_TRUNC_SEW_7, vssrl_vx_7, opivx_check_7)
GEN_OPIVI_TRANS(vssra_vi_7, IMM_TRUNC_SEW_7, vssra_vx_7, opivx_check_7)

/* Vector Narrowing Fixed-Point Clip Instructions */
GEN_OPIVV_NARROW_TRANS(vnclipu_vv_7)
GEN_OPIVV_NARROW_TRANS(vnclip_vv_7)
GEN_OPIVX_NARROW_TRANS(vnclipu_vx_7)
GEN_OPIVX_NARROW_TRANS(vnclip_vx_7)
GEN_OPIVI_NARROW_TRANS(vnclipu_vi_7, IMM_ZX, vnclipu_vx_7)
GEN_OPIVI_NARROW_TRANS(vnclip_vi_7, IMM_ZX, vnclip_vx_7)

/*
 *** Vector Float Point Arithmetic Instructions
 */
/* Vector Single-Width Floating-Point Add/Subtract Instructions */

/*
 * If the current SEW does not correspond to a supported IEEE floating-point
 * type, an illegal instruction exception is raised.
 */
static bool opfvv_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_reg(s, a->rs1, false) &&
            (s->sew != 0));
}

/* OPFVV without GVEC IR */
#undef GEN_OPFVV_TRANS
#define GEN_OPFVV_TRANS(NAME, CHECK)                               \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    if (CHECK(s, a)) {                                             \
        uint32_t data = 0;                                         \
        static gen_helper_gvec_4_ptr * const fns[3] = {            \
            gen_helper_##NAME##_h,                                 \
            gen_helper_##NAME##_w,                                 \
            gen_helper_##NAME##_d,                                 \
        };                                                         \
        TCGLabel *over = gen_new_label();                          \
        gen_set_rm(s, 7);                                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);           \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);               \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);           \
        tcg_gen_gvec_4_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),     \
                           vreg_ofs(s, a->rs1),                    \
                           vreg_ofs(s, a->rs2), cpu_env,           \
                           s->vlen / 8, s->vlen / 8, data,         \
                           fns[s->sew - 1]);                       \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}

#undef GEN_OPFVV_TRANS_B
#define GEN_OPFVV_TRANS_B(NAME, CHECK)                             \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    if (CHECK(s, a)) {                                             \
        uint32_t data = 0;                                         \
        static gen_helper_gvec_4_ptr * const fns[3] = {            \
            gen_helper_##NAME##_h,                                 \
            gen_helper_##NAME##_w,                                 \
            gen_helper_##NAME##_d,                                 \
        };                                                         \
        TCGLabel *over = gen_new_label();                          \
        gen_set_rm(s, 7);                                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);           \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);               \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);           \
        tcg_gen_gvec_4_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),     \
                           vreg_ofs(s, a->rs1),                    \
                           vreg_ofs(s, a->rs2), cpu_env,           \
                           s->vlen / 8, s->vlen / 8, data,         \
                           (s->bf16 && (s->sew == 1)) ?          \
                               gen_helper_##NAME##_bh :           \
                               fns[s->sew - 1]);                   \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}
GEN_OPFVV_TRANS_B(vfadd_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS_B(vfsub_vv_7, opfvv_check_7)

static bool opfvf_trans_7(uint32_t vd, uint32_t rs1, uint32_t vs2,
                        uint32_t data, gen_helper_opfvf *fn, DisasContext *s)
{
    TCGv_ptr dest, src2, mask;
    TCGv_i32 desc;

    TCGLabel *over = gen_new_label();
    tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

    dest = tcg_temp_new_ptr();
    mask = tcg_temp_new_ptr();
    src2 = tcg_temp_new_ptr();
    desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));

    tcg_gen_addi_ptr(dest, cpu_env, vreg_ofs(s, vd));
    tcg_gen_addi_ptr(src2, cpu_env, vreg_ofs(s, vs2));
    tcg_gen_addi_ptr(mask, cpu_env, vreg_ofs(s, 0));

    fn(dest, mask, cpu_fpr[rs1], src2, cpu_env, desc);

    gen_set_label(over);
    return true;
}

static bool opfvf_check_7(DisasContext *s, arg_rmrr *a)
{
/*
 * If the current SEW does not correspond to a supported IEEE floating-point
 * type, an illegal instruction exception is raised
 */
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            (s->sew != 0));
}

/* OPFVF without GVEC IR */
#undef GEN_OPFVF_TRANS
#define GEN_OPFVF_TRANS(NAME, CHECK)                              \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)            \
{                                                                 \
    if (CHECK(s, a)) {                                            \
        uint32_t data = 0;                                        \
        static gen_helper_opfvf *const fns[3] = {                 \
            gen_helper_##NAME##_h,                                \
            gen_helper_##NAME##_w,                                \
            gen_helper_##NAME##_d,                                \
        };                                                        \
        gen_set_rm(s, 7);                                         \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);          \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);              \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);          \
        return opfvf_trans_7(a->rd, a->rs1, a->rs2, data,         \
                                 fns[s->sew - 1], s);             \
    }                                                             \
    return false;                                                 \
}

#undef GEN_OPFVF_TRANS_B
#define GEN_OPFVF_TRANS_B(NAME, CHECK)                            \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)            \
{                                                                 \
    if (CHECK(s, a)) {                                            \
        uint32_t data = 0;                                        \
        static gen_helper_opfvf *const fns[3] = {                 \
            gen_helper_##NAME##_h,                                \
            gen_helper_##NAME##_w,                                \
            gen_helper_##NAME##_d,                                \
        };                                                        \
        gen_set_rm(s, 7);                                         \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);          \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);              \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);          \
        return opfvf_trans_7(a->rd, a->rs1, a->rs2, data,         \
                             (s->bf16 && (s->sew == 1)) ?       \
                                 gen_helper_##NAME##_bh :        \
                                 fns[s->sew - 1], s);             \
    }                                                             \
    return false;                                                 \
}
GEN_OPFVF_TRANS_B(vfadd_vf_7,  opfvf_check_7)
GEN_OPFVF_TRANS_B(vfsub_vf_7,  opfvf_check_7)
GEN_OPFVF_TRANS_B(vfrsub_vf_7,  opfvf_check_7)

/* Vector Widening Floating-Point Add/Subtract Instructions */
static bool opfvv_widen_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, true) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_reg(s, a->rs1, false) &&
            vext_check_overlap_group(a->rd, 2 << s->lmul, a->rs2,
                                     1 << s->lmul) &&
            vext_check_overlap_group(a->rd, 2 << s->lmul, a->rs1,
                                     1 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3) && (s->sew != 0));
}

/* OPFVV with WIDEN */
#undef GEN_OPFVV_WIDEN_TRANS
#define GEN_OPFVV_WIDEN_TRANS(NAME, CHECK)                       \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)           \
{                                                                \
    if (CHECK(s, a)) {                                           \
        uint32_t data = 0;                                       \
        static gen_helper_gvec_4_ptr * const fns[2] = {          \
            gen_helper_##NAME##_h, gen_helper_##NAME##_w,        \
        };                                                       \
        TCGLabel *over = gen_new_label();                        \
        gen_set_rm(s, 7);                                        \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);        \
                                                                 \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);           \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);               \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);           \
        tcg_gen_gvec_4_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),   \
                           vreg_ofs(s, a->rs1),                  \
                           vreg_ofs(s, a->rs2), cpu_env, s->vlen / 8, \
                           s->vlen / 8, data,                    \
                           (s->bf16 && (s->sew == 1)) ?        \
                               gen_helper_##NAME##_bh :         \
                               fns[s->sew - 1]);                 \
        gen_set_label(over);                                     \
        return true;                                             \
    }                                                            \
    return false;                                                \
}

GEN_OPFVV_WIDEN_TRANS(vfwadd_vv_7, opfvv_widen_check_7)
GEN_OPFVV_WIDEN_TRANS(vfwsub_vv_7, opfvv_widen_check_7)

static bool opfvf_widen_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, true) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_overlap_group(a->rd, 2 << s->lmul, a->rs2,
                                     1 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3) && (s->sew != 0));
}

/* OPFVF with WIDEN */
#undef GEN_OPFVF_WIDEN_TRANS
#define GEN_OPFVF_WIDEN_TRANS(NAME)                              \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)           \
{                                                                \
    if (opfvf_widen_check_7(s, a)) {                             \
        uint32_t data = 0;                                       \
        static gen_helper_opfvf *const fns[2] = {                \
            gen_helper_##NAME##_h, gen_helper_##NAME##_w,        \
        };                                                       \
        gen_set_rm(s, 7);                                        \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);         \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);             \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);         \
        return opfvf_trans_7(a->rd, a->rs1, a->rs2, data,        \
                             (s->bf16 && (s->sew == 1)) ?      \
                               gen_helper_##NAME##_bh :         \
                               fns[s->sew - 1], s);              \
    }                                                            \
    return false;                                                \
}

GEN_OPFVF_WIDEN_TRANS(vfwadd_vf_7)
GEN_OPFVF_WIDEN_TRANS(vfwsub_vf_7)

static bool opfwv_widen_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, true) &&
            vext_check_reg(s, a->rs2, true) &&
            vext_check_reg(s, a->rs1, false) &&
            vext_check_overlap_group(a->rd, 2 << s->lmul, a->rs1,
                                     1 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3) && (s->sew != 0));
}

/* WIDEN OPFVV with WIDEN */
#undef GEN_OPFWV_WIDEN_TRANS
#define GEN_OPFWV_WIDEN_TRANS(NAME)                                \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)             \
{                                                                  \
    if (opfwv_widen_check_7(s, a)) {                               \
        uint32_t data = 0;                                         \
        static gen_helper_gvec_4_ptr * const fns[2] = {            \
            gen_helper_##NAME##_h, gen_helper_##NAME##_w,          \
        };                                                         \
        TCGLabel *over = gen_new_label();                          \
        gen_set_rm(s, 7);                                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);             \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);                 \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);             \
        tcg_gen_gvec_4_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),     \
                           vreg_ofs(s, a->rs1),                    \
                           vreg_ofs(s, a->rs2), cpu_env, s->vlen / 8, \
                           s->vlen / 8, data,                      \
                           (s->bf16 && (s->sew == 1)) ?          \
                               gen_helper_##NAME##_bh :           \
                               fns[s->sew - 1]);                   \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}

GEN_OPFWV_WIDEN_TRANS(vfwadd_wv_7)
GEN_OPFWV_WIDEN_TRANS(vfwsub_wv_7)

static bool opfwf_widen_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, true) &&
            vext_check_reg(s, a->rs2, true) &&
            (s->lmul < 0x3) && (s->sew < 0x3) && (s->sew != 0));
}

/* WIDEN OPFVF with WIDEN */
#undef GEN_OPFWF_WIDEN_TRANS
#define GEN_OPFWF_WIDEN_TRANS(NAME)                              \
static bool trans_##NAME(DisasContext *s, arg_rmrr *a)           \
{                                                                \
    if (opfwf_widen_check_7(s, a)) {                             \
        uint32_t data = 0;                                       \
        static gen_helper_opfvf *const fns[2] = {                \
            gen_helper_##NAME##_h, gen_helper_##NAME##_w,        \
        };                                                       \
        gen_set_rm(s, 7);                                        \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);         \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);             \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);         \
        return opfvf_trans_7(a->rd, a->rs1, a->rs2, data,        \
                             (s->bf16 && (s->sew == 1)) ?      \
                               gen_helper_##NAME##_bh :         \
                               fns[s->sew - 1], s);              \
    }                                                            \
    return false;                                                \
}

GEN_OPFWF_WIDEN_TRANS(vfwadd_wf_7)
GEN_OPFWF_WIDEN_TRANS(vfwsub_wf_7)

/* Vector Single-Width Floating-Point Multiply/Divide Instructions */
GEN_OPFVV_TRANS_B(vfmul_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS_B(vfdiv_vv_7, opfvv_check_7)
GEN_OPFVF_TRANS_B(vfmul_vf_7,  opfvf_check_7)
GEN_OPFVF_TRANS_B(vfdiv_vf_7,  opfvf_check_7)
GEN_OPFVF_TRANS_B(vfrdiv_vf_7,  opfvf_check_7)

/* Vector Widening Floating-Point Multiply */
GEN_OPFVV_WIDEN_TRANS(vfwmul_vv_7, opfvv_widen_check_7)
GEN_OPFVF_WIDEN_TRANS(vfwmul_vf_7)

/* Vector Single-Width Floating-Point Fused Multiply-Add Instructions */
GEN_OPFVV_TRANS_B(vfmacc_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS_B(vfnmacc_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS_B(vfmsac_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS_B(vfnmsac_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS_B(vfmadd_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS_B(vfnmadd_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS_B(vfmsub_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS_B(vfnmsub_vv_7, opfvv_check_7)
GEN_OPFVF_TRANS_B(vfmacc_vf_7, opfvf_check_7)
GEN_OPFVF_TRANS_B(vfnmacc_vf_7, opfvf_check_7)
GEN_OPFVF_TRANS_B(vfmsac_vf_7, opfvf_check_7)
GEN_OPFVF_TRANS_B(vfnmsac_vf_7, opfvf_check_7)
GEN_OPFVF_TRANS_B(vfmadd_vf_7, opfvf_check_7)
GEN_OPFVF_TRANS_B(vfnmadd_vf_7, opfvf_check_7)
GEN_OPFVF_TRANS_B(vfmsub_vf_7, opfvf_check_7)
GEN_OPFVF_TRANS_B(vfnmsub_vf_7, opfvf_check_7)

/* Vector Widening Floating-Point Fused Multiply-Add Instructions */
GEN_OPFVV_WIDEN_TRANS(vfwmacc_vv_7, opfvv_widen_check_7)
GEN_OPFVV_WIDEN_TRANS(vfwnmacc_vv_7, opfvv_widen_check_7)
GEN_OPFVV_WIDEN_TRANS(vfwmsac_vv_7, opfvv_widen_check_7)
GEN_OPFVV_WIDEN_TRANS(vfwnmsac_vv_7, opfvv_widen_check_7)
GEN_OPFVF_WIDEN_TRANS(vfwmacc_vf_7)
GEN_OPFVF_WIDEN_TRANS(vfwnmacc_vf_7)
GEN_OPFVF_WIDEN_TRANS(vfwmsac_vf_7)
GEN_OPFVF_WIDEN_TRANS(vfwnmsac_vf_7)

/* Vector Floating-Point Square-Root Instruction */

/*
 * If the current SEW does not correspond to a supported IEEE floating-point
 * type, an illegal instruction exception is raised
 */
static bool opfv_check_7(DisasContext *s, arg_rmr *a)
{
   return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            (s->sew != 0));
}

#undef GEN_OPFV_TRANS
#define GEN_OPFV_TRANS(NAME, CHECK)                                \
static bool trans_##NAME(DisasContext *s, arg_rmr *a)              \
{                                                                  \
    if (CHECK(s, a)) {                                             \
        uint32_t data = 0;                                         \
        static gen_helper_gvec_3_ptr * const fns[3] = {            \
            gen_helper_##NAME##_h,                                 \
            gen_helper_##NAME##_w,                                 \
            gen_helper_##NAME##_d,                                 \
        };                                                         \
        TCGLabel *over = gen_new_label();                          \
        gen_set_rm(s, 7);                                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);             \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);                 \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);             \
        tcg_gen_gvec_3_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),     \
                           vreg_ofs(s, a->rs2), cpu_env, s->vlen / 8, \
                           s->vlen / 8, data,                      \
                           (s->bf16 && (s->sew == 1)) ?          \
                               gen_helper_##NAME##_bh :           \
                               fns[s->sew - 1]);                   \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}

GEN_OPFV_TRANS(vfsqrt_v_7, opfv_check_7)

/* Vector Floating-Point MIN/MAX Instructions */
GEN_OPFVV_TRANS_B(vfmin_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS_B(vfmax_vv_7, opfvv_check_7)
GEN_OPFVF_TRANS_B(vfmin_vf_7, opfvf_check_7)
GEN_OPFVF_TRANS_B(vfmax_vf_7, opfvf_check_7)

/* Vector Floating-Point Sign-Injection Instructions */
GEN_OPFVV_TRANS(vfsgnj_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS(vfsgnjn_vv_7, opfvv_check_7)
GEN_OPFVV_TRANS(vfsgnjx_vv_7, opfvv_check_7)
GEN_OPFVF_TRANS(vfsgnj_vf_7, opfvf_check_7)
GEN_OPFVF_TRANS(vfsgnjn_vf_7, opfvf_check_7)
GEN_OPFVF_TRANS(vfsgnjx_vf_7, opfvf_check_7)

/* Vector Floating-Point Compare Instructions */
static bool opfvv_cmp_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_reg(s, a->rs1, false) &&
            (s->sew != 0) &&
            ((vext_check_overlap_group(a->rd, 1, a->rs1, 1 << s->lmul) &&
              vext_check_overlap_group(a->rd, 1, a->rs2, 1 << s->lmul)) ||
             (s->lmul == 0)));
}

GEN_OPFVV_TRANS_B(vmfeq_vv_7, opfvv_cmp_check_7)
GEN_OPFVV_TRANS_B(vmfne_vv_7, opfvv_cmp_check_7)
GEN_OPFVV_TRANS_B(vmflt_vv_7, opfvv_cmp_check_7)
GEN_OPFVV_TRANS_B(vmfle_vv_7, opfvv_cmp_check_7)
GEN_OPFVV_TRANS_B(vmford_vv_7, opfvv_cmp_check_7)

static bool opfvf_cmp_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rs2, false) &&
            (s->sew != 0) &&
            (vext_check_overlap_group(a->rd, 1, a->rs2, 1 << s->lmul) ||
             (s->lmul == 0)));
}

GEN_OPFVF_TRANS_B(vmfeq_vf_7, opfvf_cmp_check_7)
GEN_OPFVF_TRANS_B(vmfne_vf_7, opfvf_cmp_check_7)
GEN_OPFVF_TRANS_B(vmflt_vf_7, opfvf_cmp_check_7)
GEN_OPFVF_TRANS_B(vmfle_vf_7, opfvf_cmp_check_7)
GEN_OPFVF_TRANS_B(vmfgt_vf_7, opfvf_cmp_check_7)
GEN_OPFVF_TRANS_B(vmfge_vf_7, opfvf_cmp_check_7)
GEN_OPFVF_TRANS_B(vmford_vf_7, opfvf_cmp_check_7)

/* Vector Floating-Point Classify Instruction */
GEN_OPFV_TRANS(vfclass_v_7, opfv_check_7)

/* Vector Floating-Point Merge Instruction */
GEN_OPFVF_TRANS(vfmerge_vfm_7,  opfvf_check_7)

static bool trans_vfmv_v_f_7(DisasContext *s, arg_vfmv_v_f_7 *a)
{
    if (vext_check_isa_ill(s) &&
        vext_check_reg(s, a->rd, false) &&
        (s->sew != 0)) {

        if (s->vl_eq_vlmax) {
            tcg_gen_gvec_dup_i64(s->sew, vreg_ofs(s, a->rd),
                                 MAXSZ_7(s), MAXSZ_7(s), cpu_fpr[a->rs1]);
        } else {
            TCGv_ptr dest;
            TCGv_i32 desc;
            uint32_t data = FIELD_DP32(0, VDATA_7, LMUL, s->lmul);
            static gen_helper_vmv_vx * const fns[3] = {
                gen_helper_vmv_v_x_7_h,
                gen_helper_vmv_v_x_7_w,
                gen_helper_vmv_v_x_7_d,
            };
            TCGLabel *over = gen_new_label();
            tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

            dest = tcg_temp_new_ptr();
            desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));
            tcg_gen_addi_ptr(dest, cpu_env, vreg_ofs(s, a->rd));
            fns[s->sew - 1](dest, cpu_fpr[a->rs1], cpu_env, desc);

            gen_set_label(over);
        }
        return true;
    }
    return false;
}

/* Single-Width Floating-Point/Integer Type-Convert Instructions */
GEN_OPFV_TRANS(vfcvt_xu_f_v_7, opfv_check_7)
GEN_OPFV_TRANS(vfcvt_x_f_v_7, opfv_check_7)
GEN_OPFV_TRANS(vfcvt_f_xu_v_7, opfv_check_7)
GEN_OPFV_TRANS(vfcvt_f_x_v_7, opfv_check_7)

/* Widening Floating-Point/Integer Type-Convert Instructions */

/*
 * If the current SEW does not correspond to a supported IEEE floating-point
 * type, an illegal instruction exception is raised
 */
static bool opfv_widen_check_7(DisasContext *s, arg_rmr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, true) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_overlap_group(a->rd, 2 << s->lmul, a->rs2,
                                     1 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3) && (s->sew != 0));
}

/*
 * If the current SEW does not correspond to a supported IEEE floating-point
 * type, an illegal instruction exception is raised
 */
static bool opfxv_widen_check_7(DisasContext *s, arg_rmr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, true) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_overlap_group(a->rd, 2 << s->lmul, a->rs2,
                                     1 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}

#undef GEN_OPFV_WIDEN_TRANS
#define GEN_OPFV_WIDEN_TRANS(NAME)                                 \
static bool trans_##NAME(DisasContext *s, arg_rmr *a)              \
{                                                                  \
    if (opfxv_widen_check_7(s, a)) {                               \
        uint32_t data = 0;                                         \
        static gen_helper_gvec_3_ptr * const fns[3] = {            \
            gen_helper_##NAME##_b,                                 \
            gen_helper_##NAME##_h,                                 \
            gen_helper_##NAME##_w,                                 \
        };                                                         \
        TCGLabel *over = gen_new_label();                          \
        gen_set_rm(s, 7);                                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);             \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);                 \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);             \
        tcg_gen_gvec_3_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),     \
                           vreg_ofs(s, a->rs2), cpu_env, s->vlen / 8, \
                           s->vlen / 8, data,                      \
                           (s->bf16 && (s->sew == 0)) ?            \
                               gen_helper_##NAME##_bb :            \
                               fns[s->sew]);                       \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}
GEN_OPFV_WIDEN_TRANS(vfwcvt_f_xu_v_7)
GEN_OPFV_WIDEN_TRANS(vfwcvt_f_x_v_7)

#undef GEN_OPFV_WIDEN_TRANS_B
#define GEN_OPFV_WIDEN_TRANS_B(NAME)                               \
static bool trans_##NAME(DisasContext *s, arg_rmr *a)              \
{                                                                  \
    if (opfv_widen_check_7(s, a)) {                                  \
        uint32_t data = 0;                                         \
        static gen_helper_gvec_3_ptr * const fns[2] = {            \
            gen_helper_##NAME##_h,                                 \
            gen_helper_##NAME##_w,                                 \
        };                                                         \
        TCGLabel *over = gen_new_label();                          \
        gen_set_rm(s, 7);                                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);             \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);                 \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);             \
        tcg_gen_gvec_3_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),     \
                           vreg_ofs(s, a->rs2), cpu_env, s->vlen / 8, \
                           s->vlen / 8, data,                      \
                           (s->bf16 && (s->sew == 1)) ?          \
                               gen_helper_##NAME##_bh :           \
                               fns[s->sew - 1]);                   \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}
GEN_OPFV_WIDEN_TRANS_B(vfwcvt_xu_f_v_7)
GEN_OPFV_WIDEN_TRANS_B(vfwcvt_x_f_v_7)
GEN_OPFV_WIDEN_TRANS_B(vfwcvt_f_f_v_7)

/* Narrowing Floating-Point/Integer Type-Convert Instructions */

/*
 * If the current SEW does not correspond to a supported IEEE floating-point
 * type, an illegal instruction exception is raised
 */
static bool opfv_narrow_check_7(DisasContext *s, arg_rmr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, true) &&
            vext_check_overlap_group(a->rd, 1 << s->lmul, a->rs2,
                                     2 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3) && (s->sew != 0));
}

/*
 * If the current SEW does not correspond to a supported IEEE floating-point
 * type, an illegal instruction exception is raised
 */
static bool opxfv_narrow_check_7(DisasContext *s, arg_rmr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, false) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, true) &&
            vext_check_overlap_group(a->rd, 1 << s->lmul, a->rs2,
                                     2 << s->lmul) &&
            (s->lmul < 0x3) && (s->sew < 0x3));
}
#undef GEN_OPFV_NARROW_TRANS
#define GEN_OPFV_NARROW_TRANS(NAME)                                \
static bool trans_##NAME(DisasContext *s, arg_rmr *a)              \
{                                                                  \
    if (opxfv_narrow_check_7(s, a)) {                              \
        uint32_t data = 0;                                         \
        static gen_helper_gvec_3_ptr * const fns[3] = {            \
            gen_helper_##NAME##_b,                                 \
            gen_helper_##NAME##_h,                                 \
            gen_helper_##NAME##_w,                                 \
        };                                                         \
        TCGLabel *over = gen_new_label();                          \
        gen_set_rm(s, 7);                                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);             \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);                 \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);             \
        tcg_gen_gvec_3_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),     \
                           vreg_ofs(s, a->rs2), cpu_env, s->vlen / 8, \
                           s->vlen / 8, data,                      \
                           (s->bf16 && (s->sew == 0)) ?            \
                               gen_helper_##NAME##_bb :            \
                               fns[s->sew]);                       \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}

GEN_OPFV_NARROW_TRANS(vfncvt_xu_f_v_7)
GEN_OPFV_NARROW_TRANS(vfncvt_x_f_v_7)

#undef GEN_OPFV_NARROW_TRANS_B
#define GEN_OPFV_NARROW_TRANS_B(NAME)                              \
static bool trans_##NAME(DisasContext *s, arg_rmr *a)              \
{                                                                  \
    if (opfv_narrow_check_7(s, a)) {                               \
        uint32_t data = 0;                                         \
        static gen_helper_gvec_3_ptr * const fns[2] = {            \
            gen_helper_##NAME##_h,                                 \
            gen_helper_##NAME##_w,                                 \
        };                                                         \
        TCGLabel *over = gen_new_label();                          \
        gen_set_rm(s, 7);                                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);             \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);                 \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);             \
        tcg_gen_gvec_3_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),     \
                           vreg_ofs(s, a->rs2), cpu_env, s->vlen / 8, \
                           s->vlen / 8, data,                      \
                           (s->bf16 && (s->sew == 1)) ?          \
                               gen_helper_##NAME##_bh :           \
                               fns[s->sew - 1]);                   \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}
GEN_OPFV_NARROW_TRANS_B(vfncvt_f_xu_v_7)
GEN_OPFV_NARROW_TRANS_B(vfncvt_f_x_v_7)
GEN_OPFV_NARROW_TRANS_B(vfncvt_f_f_v_7)

/*
 *** Vector Reduction Operations
 */
/* Vector Single-Width Integer Reduction Instructions */
static bool reduction_check_7(DisasContext *s, arg_rmrr *a)
{
    return vext_check_isa_ill(s) && vext_check_reg(s, a->rs2, false);
}

GEN_OPIVV_TRANS(vredsum_vs_7, reduction_check_7)
GEN_OPIVV_TRANS(vredmaxu_vs_7, reduction_check_7)
GEN_OPIVV_TRANS(vredmax_vs_7, reduction_check_7)
GEN_OPIVV_TRANS(vredminu_vs_7, reduction_check_7)
GEN_OPIVV_TRANS(vredmin_vs_7, reduction_check_7)
GEN_OPIVV_TRANS(vredand_vs_7, reduction_check_7)
GEN_OPIVV_TRANS(vredor_vs_7, reduction_check_7)
GEN_OPIVV_TRANS(vredxor_vs_7, reduction_check_7)

/* Vector Widening Integer Reduction Instructions */
GEN_OPIVV_WIDEN_TRANS(vwredsum_vs_7, reduction_check_7)
GEN_OPIVV_WIDEN_TRANS(vwredsumu_vs_7, reduction_check_7)

/* Vector Single-Width Floating-Point Reduction Instructions */
GEN_OPFVV_TRANS_B(vfredsum_vs_7, reduction_check_7)
GEN_OPFVV_TRANS_B(vfredmax_vs_7, reduction_check_7)
GEN_OPFVV_TRANS_B(vfredmin_vs_7, reduction_check_7)

/* Vector Widening Floating-Point Reduction Instructions */
GEN_OPFVV_WIDEN_TRANS(vfwredsum_vs_7, reduction_check_7)

/*
 *** Vector Mask Operations
 */

/* Vector Mask-Register Logical Instructions */
#undef GEN_MM_TRANS
#define GEN_MM_TRANS(NAME)                                         \
static bool trans_##NAME(DisasContext *s, arg_r *a)                \
{                                                                  \
    if (vext_check_isa_ill(s)) {                                   \
        uint32_t data = 0;                                         \
        gen_helper_gvec_4_ptr *fn = gen_helper_##NAME;             \
        TCGLabel *over = gen_new_label();                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);             \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);             \
        tcg_gen_gvec_4_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),     \
                           vreg_ofs(s, a->rs1),                    \
                           vreg_ofs(s, a->rs2), cpu_env, s->vlen / 8, \
                           s->vlen / 8, data, fn);                 \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}

GEN_MM_TRANS(vmand_mm_7)
GEN_MM_TRANS(vmnand_mm_7)
GEN_MM_TRANS(vmandnot_mm_7)
GEN_MM_TRANS(vmxor_mm_7)
GEN_MM_TRANS(vmor_mm_7)
GEN_MM_TRANS(vmnor_mm_7)
GEN_MM_TRANS(vmornot_mm_7)
GEN_MM_TRANS(vmxnor_mm_7)

/* Vector mask population count vmpopc */
static bool trans_vmpopc_m_7(DisasContext *s, arg_rmr *a)
{
    if (vext_check_isa_ill(s)) {
        TCGv_ptr src2, mask;
        TCGv dst;
        TCGv_i32 desc;
        uint32_t data = 0;
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);

        mask = tcg_temp_new_ptr();
        src2 = tcg_temp_new_ptr();
        dst = dest_gpr(s, a->rd);
        desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));

        tcg_gen_addi_ptr(src2, cpu_env, vreg_ofs(s, a->rs2));
        tcg_gen_addi_ptr(mask, cpu_env, vreg_ofs(s, 0));

        gen_helper_vmpopc_m_7(dst, mask, src2, cpu_env, desc);
        gen_set_gpr(s, a->rd, dst);

        return true;
    }
    return false;
}

/* vmfirst find-first-set mask bit */
static bool trans_vmfirst_m_7(DisasContext *s, arg_rmr *a)
{
    if (vext_check_isa_ill(s)) {
        TCGv_ptr src2, mask;
        TCGv dst;
        TCGv_i32 desc;
        uint32_t data = 0;
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);

        mask = tcg_temp_new_ptr();
        src2 = tcg_temp_new_ptr();
        dst = dest_gpr(s, a->rd);
        desc = tcg_constant_i32(simd_desc(s->vlen / 8, s->vlen / 8, data));

        tcg_gen_addi_ptr(src2, cpu_env, vreg_ofs(s, a->rs2));
        tcg_gen_addi_ptr(mask, cpu_env, vreg_ofs(s, 0));

        gen_helper_vmfirst_m_7(dst, mask, src2, cpu_env, desc);
        gen_set_gpr(s, a->rd, dst);

        return true;
    }
    return false;
}

/* vmsbf.m set-before-first mask bit */
/* vmsif.m set-includ-first mask bit */
/* vmsof.m set-only-first mask bit */
#undef GEN_M_TRANS
#define GEN_M_TRANS(NAME)                                          \
static bool trans_##NAME(DisasContext *s, arg_rmr *a)              \
{                                                                  \
    if (vext_check_isa_ill(s)) {                                   \
        uint32_t data = 0;                                         \
        gen_helper_gvec_3_ptr *fn = gen_helper_##NAME;             \
        TCGLabel *over = gen_new_label();                          \
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);          \
                                                                   \
        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);             \
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);                 \
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);             \
        tcg_gen_gvec_3_ptr(vreg_ofs(s, a->rd),                     \
                           vreg_ofs(s, 0), vreg_ofs(s, a->rs2),    \
                           cpu_env, s->vlen / 8, s->vlen / 8, data,\
                           fn);                                    \
        gen_set_label(over);                                       \
        return true;                                               \
    }                                                              \
    return false;                                                  \
}

GEN_M_TRANS(vmsbf_m_7)
GEN_M_TRANS(vmsif_m_7)
GEN_M_TRANS(vmsof_m_7)

/* Vector Iota Instruction */
static bool trans_viota_m_7(DisasContext *s, arg_viota_m *a)
{
    if (vext_check_isa_ill(s) &&
        vext_check_reg(s, a->rd, false) &&
        vext_check_overlap_group(a->rd, 1 << s->lmul, a->rs2, 1) &&
        (a->vm != 0 || a->rd != 0)) {
        uint32_t data = 0;
        TCGLabel *over = gen_new_label();
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
        static gen_helper_gvec_3_ptr * const fns[4] = {
            gen_helper_viota_m_7_b, gen_helper_viota_m_7_h,
            gen_helper_viota_m_7_w, gen_helper_viota_m_7_d,
        };
        tcg_gen_gvec_3_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),
                           vreg_ofs(s, a->rs2), cpu_env, s->vlen / 8,
                           s->vlen / 8, data, fns[s->sew]);
        gen_set_label(over);
        return true;
    }
    return false;
}

/* Vector Element Index Instruction */
static bool trans_vid_v_7(DisasContext *s, arg_vid_v_7 *a)
{
    if (vext_check_isa_ill(s) &&
        vext_check_reg(s, a->rd, false) &&
        vext_check_overlap_mask(s, a->rd, a->vm, false)) {
        uint32_t data = 0;
        TCGLabel *over = gen_new_label();
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
        data = FIELD_DP32(data, VDATA_7, VM, a->vm);
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
        static gen_helper_gvec_2_ptr * const fns[4] = {
            gen_helper_vid_v_7_b, gen_helper_vid_v_7_h,
            gen_helper_vid_v_7_w, gen_helper_vid_v_7_d,
        };
        tcg_gen_gvec_2_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),
                           cpu_env, s->vlen / 8, s->vlen / 8, data, fns[s->sew]);
        gen_set_label(over);
        return true;
    }
    return false;
}

/*
 *** Vector Permutation Instructions
 */

/* Integer Extract Instruction */

static void load_element_7(TCGv_i64 dest, TCGv_ptr base,
                         int ofs, int sew)
{
    switch (sew) {
    case MO_8:
        tcg_gen_ld8u_i64(dest, base, ofs);
        break;
    case MO_16:
        tcg_gen_ld16u_i64(dest, base, ofs);
        break;
    case MO_32:
        tcg_gen_ld32u_i64(dest, base, ofs);
        break;
    case MO_64:
        tcg_gen_ld_i64(dest, base, ofs);
        break;
    default:
        g_assert_not_reached();
        break;
    }
}

/* Load idx >= VLMAX ? 0 : vreg[idx] */
static void vec_element_loadx_7(DisasContext *s, TCGv_i64 dest,
                              int vreg, TCGv idx, int vlmax)
{
    TCGv_i32 ofs = tcg_temp_new_i32();
    TCGv_ptr base = tcg_temp_new_ptr();
    TCGv_i64 t_idx = tcg_temp_new_i64();
    TCGv_i64 t_vlmax, t_zero;

    /*
     * Mask the index to the length so that we do
     * not produce an out-of-range load.
     */
    tcg_gen_trunc_tl_i32(ofs, idx);
    tcg_gen_andi_i32(ofs, ofs, vlmax - 1);

    /* Convert the index to an offset. */
    endian_adjust(ofs, s->sew);
    tcg_gen_shli_i32(ofs, ofs, s->sew);

    /* Convert the index to a pointer. */
    tcg_gen_ext_i32_ptr(base, ofs);
    tcg_gen_add_ptr(base, base, cpu_env);

    /* Perform the load. */
    load_element_7(dest, base,
                 vreg_ofs(s, vreg), s->sew);

    /* Flush out-of-range indexing to zero.  */
    t_vlmax = tcg_constant_i64(vlmax);
    t_zero = tcg_constant_i64(0);
    tcg_gen_extu_tl_i64(t_idx, idx);

    tcg_gen_movcond_i64(TCG_COND_LTU, dest, t_idx,
        t_vlmax, dest, t_zero);
}

static void vec_element_loadi_7(DisasContext *s, TCGv_i64 dest,
                              int vreg, int idx)
{
    load_element_7(dest, cpu_env, endian_ofs(s, vreg, idx), s->sew);
}

static bool trans_vext_x_v_7(DisasContext *s, arg_r *a)
{
    TCGv_i64 tmp = tcg_temp_new_i64();
    TCGv dest = dest_gpr(s, a->rd);

    if (a->rs1 == 0) {
        /* Special case vmv.x.s rd, vs2. */
        vec_element_loadi_7(s, tmp, a->rs2, 0);
    } else {
        /* This instruction ignores LMUL and vector register groups */
        int vlmax = s->vlen >> (3 + s->sew);
        vec_element_loadx_7(s, tmp, a->rs2, cpu_gpr[a->rs1], vlmax);
    }

    tcg_gen_trunc_i64_tl(dest, tmp);
    gen_set_gpr(s, a->rd, dest);

    return true;
}

/* Integer Scalar Move Instruction */

/* vmv.s.x vd, rs1 # vd[0] = rs1 */
static bool trans_vmv_s_x_7(DisasContext *s, arg_vmv_s_x_7 *a)
{
    if (vext_check_isa_ill(s)) {
        /* This instruction ignores LMUL and vector register groups */
        int maxsz = s->vlen >> 3;
        TCGv_i64 t1;
        TCGLabel *over = gen_new_label();
        TCGv src1 = get_gpr(s, a->rs1, EXT_ZERO);

        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);
        tcg_gen_gvec_dup_imm(SEW64, vreg_ofs(s, a->rd), maxsz, maxsz, 0);
        if (a->rs1 == 0) {
            goto done;
        }

        t1 = tcg_temp_new_i64();
        tcg_gen_extu_tl_i64(t1, src1);
        vec_element_storei(s, a->rd, 0, t1);
    done:
        gen_set_label(over);
        return true;
    }
    return false;
}

/* Floating-Point Scalar Move Instructions */
static bool trans_vfmv_f_s_7(DisasContext *s, arg_vfmv_f_s_7 *a)
{
    if (!s->vill && has_ext(s, RVF) &&
        (s->mstatus_fs != 0) && (s->sew != 0)) {
        unsigned int len = 8 << s->sew;

        vec_element_loadi_7(s, cpu_fpr[a->rd], a->rs2, 0);
        if (len < 64) {
            tcg_gen_ori_i64(cpu_fpr[a->rd], cpu_fpr[a->rd],
                            MAKE_64BIT_MASK(len, 64 - len));
        }

        mark_fs_dirty(s);
        return true;
    }
    return false;
}

/* vfmv.s.f vd, rs1 # vd[0] = rs1 (vs2=0) */
static bool trans_vfmv_s_f_7(DisasContext *s, arg_vfmv_s_f_7 *a)
{
    if (!s->vill && has_ext(s, RVF) && (s->sew != 0)) {
        TCGv_i64 t1;
        /* The instructions ignore LMUL and vector register group. */
        uint32_t vlmax = s->vlen >> 3;

        /* if vl == 0, skip vector register write back */
        TCGLabel *over = gen_new_label();
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

        /* zeroed all elements */
        tcg_gen_gvec_dup_imm(SEW64, vreg_ofs(s, a->rd), vlmax, vlmax, 0);

        /* NaN-box f[rs1] as necessary for SEW */
        t1 = tcg_temp_new_i64();
        if (s->sew == MO_64 && !has_ext(s, RVD)) {
            tcg_gen_ori_i64(t1, cpu_fpr[a->rs1], MAKE_64BIT_MASK(32, 32));
        } else {
            tcg_gen_mov_i64(t1, cpu_fpr[a->rs1]);
        }
        vec_element_storei(s, a->rd, 0, t1);
        gen_set_label(over);
        return true;
    }
    return false;
}

/* Vector Slide Instructions */
static bool slideup_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            (a->rd != a->rs2));
}

GEN_OPIVX_TRANS(vslideup_vx_7, slideup_check_7)
GEN_OPIVX_TRANS(vslide1up_vx_7, slideup_check_7)
GEN_OPIVI_TRANS(vslideup_vi_7, IMM_ZX_7, vslideup_vx_7, slideup_check_7)

GEN_OPIVX_TRANS(vslidedown_vx_7, opivx_check_7)
GEN_OPIVX_TRANS(vslide1down_vx_7, opivx_check_7)
GEN_OPIVI_TRANS(vslidedown_vi_7, IMM_ZX_7, vslidedown_vx_7, opivx_check_7)

/* Vector Register Gather Instruction */
static bool vrgather_vv_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs1, false) &&
            vext_check_reg(s, a->rs2, false) &&
            (a->rd != a->rs2) && (a->rd != a->rs1));
}

GEN_OPIVV_TRANS(vrgather_vv_7, vrgather_vv_check_7)

static bool vrgather_vx_check_7(DisasContext *s, arg_rmrr *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_overlap_mask(s, a->rd, a->vm, true) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            (a->rd != a->rs2));
}

/* vrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] >= VLMAX) ? 0 : vs2[rs1] */
static bool trans_vrgather_vx_7(DisasContext *s, arg_rmrr *a)
{
    if (!vrgather_vx_check_7(s, a)) {
        return false;
    }

    if (a->vm && s->vl_eq_vlmax) {
        int vlmax = s->vlen / s->mlen;
        TCGv_i64 dest = tcg_temp_new_i64();

        if (a->rs1 == 0) {
            vec_element_loadi_7(s, dest, a->rs2, 0);
        } else {
            vec_element_loadx_7(s, dest, a->rs2, cpu_gpr[a->rs1], vlmax);
        }

        tcg_gen_gvec_dup_i64(s->sew, vreg_ofs(s, a->rd),
                             MAXSZ_7(s), MAXSZ_7(s), dest);
    } else {
        static gen_helper_opivx * const fns[4] = {
            gen_helper_vrgather_vx_7_b, gen_helper_vrgather_vx_7_h,
            gen_helper_vrgather_vx_7_w, gen_helper_vrgather_vx_7_d
        };
        return opivx_trans_7(a->rd, a->rs1, a->rs2, a->vm, fns[s->sew], s);
    }
    return true;
}

/* vrgather.vi vd, vs2, imm, vm # vd[i] = (imm >= VLMAX) ? 0 : vs2[imm] */
static bool trans_vrgather_vi_7(DisasContext *s, arg_rmrr *a)
{
    if (!vrgather_vx_check_7(s, a)) {
        return false;
    }

    if (a->vm && s->vl_eq_vlmax) {
        if (a->rs1 >= s->vlen / s->mlen) {
            tcg_gen_gvec_dup_imm(SEW64, vreg_ofs(s, a->rd),
                                 MAXSZ_7(s), MAXSZ_7(s), 0);
        } else {
            tcg_gen_gvec_dup_mem(s->sew, vreg_ofs(s, a->rd),
                                 endian_ofs(s, a->rs2, a->rs1),
                                 MAXSZ_7(s), MAXSZ_7(s));
        }
    } else {
        static gen_helper_opivx * const fns[4] = {
            gen_helper_vrgather_vx_7_b, gen_helper_vrgather_vx_7_h,
            gen_helper_vrgather_vx_7_w, gen_helper_vrgather_vx_7_d
        };
        return opivi_trans_7(a->rd, a->rs1, a->rs2, a->vm, fns[s->sew], s, IMM_ZX);
    }
    return true;
}

/* Vector Compress Instruction */
static bool vcompress_vm_check_7(DisasContext *s, arg_r *a)
{
    return (vext_check_isa_ill(s) &&
            vext_check_reg(s, a->rd, false) &&
            vext_check_reg(s, a->rs2, false) &&
            vext_check_overlap_group(a->rd, 1 << s->lmul, a->rs1, 1) &&
            (a->rd != a->rs2));
}

static bool trans_vcompress_vm_7(DisasContext *s, arg_r *a)
{
    if (vcompress_vm_check_7(s, a)) {
        uint32_t data = 0;
        static gen_helper_gvec_4_ptr * const fns[4] = {
            gen_helper_vcompress_vm_7_b, gen_helper_vcompress_vm_7_h,
            gen_helper_vcompress_vm_7_w, gen_helper_vcompress_vm_7_d,
        };
        TCGLabel *over = gen_new_label();
        tcg_gen_brcondi_tl(TCG_COND_EQ, cpu_vl, 0, over);

        data = FIELD_DP32(data, VDATA_7, MLEN, s->mlen);
        data = FIELD_DP32(data, VDATA_7, LMUL, s->lmul);
        tcg_gen_gvec_4_ptr(vreg_ofs(s, a->rd), vreg_ofs(s, 0),
                           vreg_ofs(s, a->rs1), vreg_ofs(s, a->rs2),
                           cpu_env, s->vlen / 8, s->vlen / 8, data, fns[s->sew]);
        gen_set_label(over);
        return true;
    }
    return false;
}
