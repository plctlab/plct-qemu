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
