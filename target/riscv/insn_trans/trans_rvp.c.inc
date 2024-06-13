/*
 * RISC-V translation routines for the P Standard Extensions.
 *
 * Copyright (c) 2020-2023 PLCT Lab
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

#define REQUIRE_ZBPBO(ctx) do {             \
    if (!ctx->cfg_ptr->ext_zbpbo) {         \
        return false;                       \
    }                                       \
} while (0)

#define REQUIRE_ZPSFOPERAND(ctx) do {       \
    if (!ctx->cfg_ptr->ext_zpsfoperand) {   \
        return false;                       \
    }                                       \
} while (0)

#define REQUIRE_ZPN(ctx) do {               \
    if (!ctx->cfg_ptr->ext_zpn) {           \
        return false;                       \
    }                                       \
} while (0)

/*
 *  Create translation entrance for instruction in Zpn
 *  sub-extension that:
 *  1. instruction format is @r (in insn32.decode)
 *  2. available in both RV32 and RV64
 */
#define GEN_SIMD_TRANS(NAME)                                \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_ZPN(ctx);                                       \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, src1, src2);                    \
    gen_set_gpr(ctx, a->rd, dest);                          \
    return true;                                            \
}

GEN_SIMD_TRANS(add16)
GEN_SIMD_TRANS(radd16)
GEN_SIMD_TRANS(uradd16)
GEN_SIMD_TRANS(kadd16)
GEN_SIMD_TRANS(ukadd16)

GEN_SIMD_TRANS(sub16)
GEN_SIMD_TRANS(rsub16)
GEN_SIMD_TRANS(ursub16)
GEN_SIMD_TRANS(ksub16)
GEN_SIMD_TRANS(uksub16)

GEN_SIMD_TRANS(cras16)
GEN_SIMD_TRANS(rcras16)
GEN_SIMD_TRANS(urcras16)
GEN_SIMD_TRANS(kcras16)
GEN_SIMD_TRANS(ukcras16)

GEN_SIMD_TRANS(crsa16)
GEN_SIMD_TRANS(rcrsa16)
GEN_SIMD_TRANS(urcrsa16)
GEN_SIMD_TRANS(kcrsa16)
GEN_SIMD_TRANS(ukcrsa16)

GEN_SIMD_TRANS(stas16)
GEN_SIMD_TRANS(rstas16)
GEN_SIMD_TRANS(urstas16)
GEN_SIMD_TRANS(kstas16)
GEN_SIMD_TRANS(ukstas16)

GEN_SIMD_TRANS(stsa16)
GEN_SIMD_TRANS(rstsa16)
GEN_SIMD_TRANS(urstsa16)
GEN_SIMD_TRANS(kstsa16)
GEN_SIMD_TRANS(ukstsa16)

GEN_SIMD_TRANS(add8)
GEN_SIMD_TRANS(radd8)
GEN_SIMD_TRANS(uradd8)
GEN_SIMD_TRANS(kadd8)
GEN_SIMD_TRANS(ukadd8)

GEN_SIMD_TRANS(sub8)
GEN_SIMD_TRANS(rsub8)
GEN_SIMD_TRANS(ursub8)
GEN_SIMD_TRANS(ksub8)
GEN_SIMD_TRANS(uksub8)

GEN_SIMD_TRANS(sra16)

static bool trans_srai16(DisasContext *ctx, arg_srai16 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_srai16(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(sra16_u)

static bool trans_srai16_u(DisasContext *ctx, arg_srai16_u *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_srai16_u(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(srl16)

static bool trans_srli16(DisasContext *ctx, arg_srli16 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_srli16(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(srl16_u)

static bool trans_srli16_u(DisasContext *ctx, arg_srli16_u *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_srli16_u(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(sll16)

static bool trans_slli16(DisasContext *ctx, arg_slli16 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_slli16(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(ksll16)

static bool trans_kslli16(DisasContext *ctx, arg_kslli16 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_kslli16(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(kslra16)
GEN_SIMD_TRANS(kslra16_u)

GEN_SIMD_TRANS(sra8)

static bool trans_srai8(DisasContext *ctx, arg_srai8 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_srai8(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(sra8_u)

static bool trans_srai8_u(DisasContext *ctx, arg_srai8_u *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_srai8_u(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(srl8)

static bool trans_srli8(DisasContext *ctx, arg_srli8 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_srli8(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(srl8_u)

static bool trans_srli8_u(DisasContext *ctx, arg_srli8_u *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_srli8_u(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(sll8)

static bool trans_slli8(DisasContext *ctx, arg_slli8 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_slli8(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(ksll8)

static bool trans_kslli8(DisasContext *ctx, arg_kslli8 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_kslli8(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(kslra8)
GEN_SIMD_TRANS(kslra8_u)

GEN_SIMD_TRANS(cmpeq16)
GEN_SIMD_TRANS(scmplt16)
GEN_SIMD_TRANS(scmple16)
GEN_SIMD_TRANS(ucmplt16)
GEN_SIMD_TRANS(ucmple16)

GEN_SIMD_TRANS(cmpeq8)
GEN_SIMD_TRANS(scmplt8)
GEN_SIMD_TRANS(scmple8)
GEN_SIMD_TRANS(ucmplt8)
GEN_SIMD_TRANS(ucmple8)

static bool trans_smul16(DisasContext *ctx, arg_smul16 *a)
{
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);

#if defined(TARGET_RISCV32)
    size_t rd = a->rd >> 1 << 1;
    TCGv dest1 = dest_gpr(ctx, rd);
    TCGv dest2 = dest_gpr(ctx, rd + 1);
    TCGv t1 = tcg_temp_new();
    TCGv t2 = tcg_temp_new();
    
    tcg_gen_sextract_tl(t1, src1, 0, 16);
    tcg_gen_sextract_tl(t2, src2, 0, 16);
    tcg_gen_mul_tl(dest1, t1, t2);
    tcg_gen_sextract_tl(t1, src1, 16, 16);
    tcg_gen_sextract_tl(t2, src2, 16, 16);
    tcg_gen_mul_tl(dest2, t1, t2);
    gen_set_gpr(ctx, rd, dest1);
    gen_set_gpr(ctx, rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_smul16_64(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
#endif

    return true;
}

static bool trans_smulx16(DisasContext *ctx, arg_smulx16 *a)
{
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);

#if defined(TARGET_RISCV32)
    size_t rd = a->rd >> 1 << 1;
    TCGv dest1 = dest_gpr(ctx, rd);
    TCGv dest2 = dest_gpr(ctx, rd + 1);
    TCGv t1 = tcg_temp_new();
    TCGv t2 = tcg_temp_new();

    tcg_gen_sextract_tl(t1, src1, 0, 16);
    tcg_gen_sextract_tl(t2, src2, 16, 16);
    tcg_gen_mul_tl(dest1, t1, t2);
    tcg_gen_sextract_tl(t1, src1, 16, 16);
    tcg_gen_sextract_tl(t2, src2, 0, 16);
    tcg_gen_mul_tl(dest2, t1, t2);
    gen_set_gpr(ctx, a->rd, dest1);
    gen_set_gpr(ctx, a->rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_smulx16_64(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
#endif

    return true;
}


static bool trans_umul16(DisasContext *ctx, arg_umul16 *a)
{
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);

#if defined(TARGET_RISCV32)
    size_t rd = a->rd >> 1 << 1;
    TCGv dest1 = dest_gpr(ctx, rd);
    TCGv dest2 = dest_gpr(ctx, rd + 1);
    TCGv t1 = tcg_temp_new();
    TCGv t2 = tcg_temp_new();

    tcg_gen_extract_tl(t1, src1, 0, 16);
    tcg_gen_extract_tl(t2, src2, 0, 16);
    tcg_gen_mul_tl(dest1, t1, t2);
    tcg_gen_extract_tl(t1, src1, 16, 16);
    tcg_gen_extract_tl(t2, src2, 16, 16);
    tcg_gen_mul_tl(dest2, t1, t2);
    gen_set_gpr(ctx, rd, dest1);
    gen_set_gpr(ctx, rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_umul16_64(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
#endif

    return true;
}

static bool trans_umulx16(DisasContext *ctx, arg_umulx16 *a)
{
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);

#if defined(TARGET_RISCV32)
    size_t rd = a->rd >> 1 << 1;
    TCGv dest1 = dest_gpr(ctx, rd);
    TCGv dest2 = dest_gpr(ctx, rd + 1);
    TCGv t1 = tcg_temp_new();
    TCGv t2 = tcg_temp_new();

    tcg_gen_extract_tl(t1, src1, 0, 16);
    tcg_gen_extract_tl(t2, src2, 16, 16);
    tcg_gen_mul_tl(dest1, t1, t2);
    tcg_gen_extract_tl(t1, src1, 16, 16);
    tcg_gen_extract_tl(t2, src2, 0, 16);
    tcg_gen_mul_tl(dest2, t1, t2);
    gen_set_gpr(ctx, rd, dest1);
    gen_set_gpr(ctx, rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_umulx16_64(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
#endif

    return true;
}

GEN_SIMD_TRANS(khm16)
GEN_SIMD_TRANS(khmx16)

static bool trans_smul8(DisasContext *ctx, arg_smul8 *a)
{
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);

#if defined(TARGET_RISCV32)
    size_t rd = a->rd >> 1 << 1;
    TCGv dest1 = dest_gpr(ctx, rd);
    TCGv dest2 = dest_gpr(ctx, rd + 1);
    TCGv t1 = tcg_temp_new();
    TCGv t2 = tcg_temp_new();
    
    tcg_gen_sextract_tl(t1, src1, 0, 8);
    tcg_gen_sextract_tl(t2, src2, 0, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest1, dest1, t1, 0, 16);
    tcg_gen_sextract_tl(t1, src1, 8, 8);
    tcg_gen_sextract_tl(t2, src2, 8, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest1, dest1, t1, 16, 16);

    tcg_gen_sextract_tl(t1, src1, 16, 8);
    tcg_gen_sextract_tl(t2, src2, 16, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest2, dest2, t1, 0, 16);
    tcg_gen_sextract_tl(t1, src1, 24, 8);
    tcg_gen_sextract_tl(t2, src2, 24, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest2, dest2, t1, 16, 16);

    gen_set_gpr(ctx, rd, dest1);
    gen_set_gpr(ctx, rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_smul8_64(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
#endif

    return true;
}

static bool trans_smulx8(DisasContext *ctx, arg_smulx8 *a)
{
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);

#if defined(TARGET_RISCV32)
    size_t rd = a->rd >> 1 << 1;
    TCGv dest1 = dest_gpr(ctx, rd);
    TCGv dest2 = dest_gpr(ctx, rd + 1);
    TCGv t1 = tcg_temp_new();
    TCGv t2 = tcg_temp_new();
    
    tcg_gen_sextract_tl(t1, src1, 0, 8);
    tcg_gen_sextract_tl(t2, src2, 8, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest1, dest1, t1, 0, 16);
    tcg_gen_sextract_tl(t1, src1, 8, 8);
    tcg_gen_sextract_tl(t2, src2, 0, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest1, dest1, t1, 16, 16);

    tcg_gen_sextract_tl(t1, src1, 16, 8);
    tcg_gen_sextract_tl(t2, src2, 24, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest2, dest2, t1, 0, 16);
    tcg_gen_sextract_tl(t1, src1, 24, 8);
    tcg_gen_sextract_tl(t2, src2, 16, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest2, dest2, t1, 16, 16);

    gen_set_gpr(ctx, rd, dest1);
    gen_set_gpr(ctx, rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_smulx8_64(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
#endif

    return true;
}


static bool trans_umul8(DisasContext *ctx, arg_umul8 *a)
{
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);

#if defined(TARGET_RISCV32)
    size_t rd = a->rd >> 1 << 1;
    TCGv dest1 = dest_gpr(ctx, rd);
    TCGv dest2 = dest_gpr(ctx, rd + 1);
    TCGv t1 = tcg_temp_new();
    TCGv t2 = tcg_temp_new();
    
    tcg_gen_extract_tl(t1, src1, 0, 8);
    tcg_gen_extract_tl(t2, src2, 0, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest1, dest1, t1, 0, 16);
    tcg_gen_extract_tl(t1, src1, 8, 8);
    tcg_gen_extract_tl(t2, src2, 8, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest1, dest1, t1, 16, 16);

    tcg_gen_extract_tl(t1, src1, 16, 8);
    tcg_gen_extract_tl(t2, src2, 16, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest2, dest2, t1, 0, 16);
    tcg_gen_extract_tl(t1, src1, 24, 8);
    tcg_gen_extract_tl(t2, src2, 24, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest2, dest2, t1, 16, 16);

    gen_set_gpr(ctx, rd, dest1);
    gen_set_gpr(ctx, rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_umul8_64(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
#endif

    return true;
}

static bool trans_umulx8(DisasContext *ctx, arg_umulx8 *a)
{
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);

#if defined(TARGET_RISCV32)
    size_t rd = a->rd >> 1 << 1;
    TCGv dest1 = dest_gpr(ctx, rd);
    TCGv dest2 = dest_gpr(ctx, rd + 1);
    TCGv t1 = tcg_temp_new();
    TCGv t2 = tcg_temp_new();
    
    tcg_gen_extract_tl(t1, src1, 0, 8);
    tcg_gen_extract_tl(t2, src2, 8, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest1, dest1, t1, 0, 16);
    tcg_gen_extract_tl(t1, src1, 8, 8);
    tcg_gen_extract_tl(t2, src2, 0, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest1, dest1, t1, 16, 16);

    tcg_gen_extract_tl(t1, src1, 16, 8);
    tcg_gen_extract_tl(t2, src2, 24, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest2, dest2, t1, 0, 16);
    tcg_gen_extract_tl(t1, src1, 24, 8);
    tcg_gen_extract_tl(t2, src2, 16, 8);
    tcg_gen_mul_tl(t1, t1, t2);
    tcg_gen_deposit_tl(dest2, dest2, t1, 16, 16);

    gen_set_gpr(ctx, rd, dest1);
    gen_set_gpr(ctx, rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_umulx8_64(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
#endif

    return true;
}

GEN_SIMD_TRANS(khm8)
GEN_SIMD_TRANS(khmx8)

GEN_SIMD_TRANS(smin16)
GEN_SIMD_TRANS(umin16)
GEN_SIMD_TRANS(smax16)
GEN_SIMD_TRANS(umax16)

static bool trans_sclip16(DisasContext *ctx, arg_sclip16 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_sclip16(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_uclip16(DisasContext *ctx, arg_uclip16 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_uclip16(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_kabs16(DisasContext *ctx, arg_kabs16 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_kabs16(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_clrs16(DisasContext *ctx, arg_clrs16 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_clrs16(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_clz16(DisasContext *ctx, arg_clz16 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_clz16(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(smin8)
GEN_SIMD_TRANS(umin8)
GEN_SIMD_TRANS(smax8)
GEN_SIMD_TRANS(umax8)

static bool trans_sclip8(DisasContext *ctx, arg_sclip8 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_sclip8(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_uclip8(DisasContext *ctx, arg_uclip8 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_uclip8(dest, src1, shamt);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_kabs8(DisasContext *ctx, arg_kabs8 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_kabs8(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_clrs8(DisasContext *ctx, arg_clrs8 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_clrs8(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_clz8(DisasContext *ctx, arg_clz8 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_clz8(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_sunpkd810(DisasContext *ctx, arg_sunpkd810 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_sunpkd810(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_sunpkd820(DisasContext *ctx, arg_sunpkd820 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_sunpkd820(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_sunpkd830(DisasContext *ctx, arg_sunpkd830 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_sunpkd830(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_sunpkd831(DisasContext *ctx, arg_sunpkd831 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_sunpkd831(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_sunpkd832(DisasContext *ctx, arg_sunpkd832 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_sunpkd832(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_zunpkd810(DisasContext *ctx, arg_zunpkd810 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_zunpkd810(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_zunpkd820(DisasContext *ctx, arg_zunpkd820 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_zunpkd820(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_zunpkd830(DisasContext *ctx, arg_zunpkd830 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_zunpkd830(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_zunpkd831(DisasContext *ctx, arg_zunpkd831 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_zunpkd831(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_zunpkd832(DisasContext *ctx, arg_zunpkd832 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_zunpkd832(dest, src1);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_pkbb16(DisasContext *ctx, arg_pkbb16 *a)
{
    REQUIRE_64BIT(ctx);
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_pkbb16(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(pkbt16)
GEN_SIMD_TRANS(pktb16)

static bool trans_pktt16(DisasContext *ctx, arg_pkbb16 *a)
{
    REQUIRE_64BIT(ctx);
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_pktt16(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_smmul(DisasContext *ctx, arg_smmul *a)
{
    REQUIRE_64BIT(ctx);
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_smmul(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

GEN_SIMD_TRANS(smmul_u)

static bool trans_kmmac(DisasContext *ctx, arg_kmmac *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmac(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmmac_u(DisasContext *ctx, arg_kmmac_u *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmac_u(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmmsb(DisasContext *ctx, arg_kmmsb *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmsb(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmmsb_u(DisasContext *ctx, arg_kmmsb_u *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmsb_u(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

GEN_SIMD_TRANS(kwmmul)
GEN_SIMD_TRANS(kwmmul_u)

GEN_SIMD_TRANS(smmwb)
GEN_SIMD_TRANS(smmwb_u)
GEN_SIMD_TRANS(smmwt)
GEN_SIMD_TRANS(smmwt_u)

static bool trans_kmmawb(DisasContext *ctx, arg_kmmawb *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmawb(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmmawb_u(DisasContext *ctx, arg_kmmawb_u *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmawb_u(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmmawt(DisasContext *ctx, arg_kmmawt *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmawt(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmmawt_u(DisasContext *ctx, arg_kmmawt_u *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmawt_u(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

GEN_SIMD_TRANS(kmmwb2)
GEN_SIMD_TRANS(kmmwb2_u)
GEN_SIMD_TRANS(kmmwt2)
GEN_SIMD_TRANS(kmmwt2_u)

static bool trans_kmmawb2(DisasContext *ctx, arg_kmmawb2 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmawb2(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmmawb2_u(DisasContext *ctx, arg_kmmawb2_u *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmawb2_u(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmmawt2(DisasContext *ctx, arg_kmmawb2 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmawt2(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmmawt2_u(DisasContext *ctx, arg_kmmawb2_u *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmmawt2_u(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

GEN_SIMD_TRANS(smbb16)
GEN_SIMD_TRANS(smbt16)
GEN_SIMD_TRANS(smtt16)
GEN_SIMD_TRANS(kmda)
GEN_SIMD_TRANS(kmxda)
GEN_SIMD_TRANS(smds)
GEN_SIMD_TRANS(smdrs)
GEN_SIMD_TRANS(smxds)

static bool trans_kmabb(DisasContext *ctx, arg_kmabb *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmabb(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmabt(DisasContext *ctx, arg_kmabt *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmabt(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmatt(DisasContext *ctx, arg_kmatt *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmatt(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmada(DisasContext *ctx, arg_kmada *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmada(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmaxda(DisasContext *ctx, arg_kmaxda *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmaxda(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmads(DisasContext *ctx, arg_kmads *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmads(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmadrs(DisasContext *ctx, arg_kmadrs *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmadrs(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmaxds(DisasContext *ctx, arg_kmaxds *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmaxds(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmsda(DisasContext *ctx, arg_kmsda *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmsda(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_kmsxda(DisasContext *ctx, arg_kmsxda *a)
{
    REQUIRE_ZPN(ctx);
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv t = tcg_temp_new();
    gen_helper_kmsxda(t, src1, src2, dest);
    gen_set_gpr(ctx, a->rd, t);
    return true;
}

static bool trans_smal(DisasContext *ctx, arg_smal *a)
{
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);

#if defined(TARGET_RISCV32)
    size_t rs1 = a->rs1 >> 1 << 1;
    size_t rd = a->rd >> 1 << 1;
    TCGv_i64 rs1_64 = tcg_temp_new_i64();
    TCGv_i64 rs2_64 = tcg_temp_new_i64();
    TCGv_i64 rd_64 = tcg_temp_new_i64();
    TCGv_i64 rs2_h = tcg_temp_new_i64();
    TCGv_i64 rs2_l = tcg_temp_new_i64();
    TCGv_i64 rs2_mul = tcg_temp_new_i64();
    TCGv_i32 Idx2 = tcg_temp_new();
    TCGv_i32 Idx3 = tcg_temp_new();
    TCGv Idx0 = dest_gpr(ctx, rs1);
    TCGv Idx1 = dest_gpr(ctx, rs1 + 1);
 
    tcg_gen_ext_i32_i64(rs2_64, src2);
    tcg_gen_sextract_i64(rs2_h, rs2_64, 16, 16);
    tcg_gen_sextract_i64(rs2_l, rs2_64, 0, 16);
    tcg_gen_mul_i64(rs2_mul, rs2_h, rs2_l);
    tcg_gen_concat_i32_i64(rs1_64, Idx0, Idx1);
    tcg_gen_add_i64(rd_64, rs1_64, rs2_mul);

    tcg_gen_trunc_i64_tl(Idx2, rd_64);
    tcg_gen_extract_i64(rd_64, rd_64, 32, 32);
    tcg_gen_trunc_i64_tl(Idx3, rd_64);
    gen_set_gpr(ctx, rd, Idx2);
    gen_set_gpr(ctx, rd + 1, Idx3);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv src1 = dest_gpr(ctx, a->rs1);
    gen_helper_smal_64(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
#endif

    return true;
}
