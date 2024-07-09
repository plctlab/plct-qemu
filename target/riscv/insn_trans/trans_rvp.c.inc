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

/* Concat two 32 bit data in src and src + 1 to dst */
static void get_pair_regs(DisasContext *ctx, TCGv_i64 dst, int src)
{
#if defined(TARGET_RISCV32)
    TCGv t1 = get_gpr(ctx, src, EXT_NONE);      // low
    TCGv t2 = get_gpr(ctx, src + 1, EXT_NONE);  // high
    tcg_gen_concat_i32_i64(dst, t1, t2);
#else
    TCGv t1 = get_gpr(ctx, src, EXT_NONE);
    tcg_gen_mov_tl(dst, t1);
#endif
}

/* Save a 64 bit data in src to dst and dst + 1 */
static void set_pair_regs(DisasContext *ctx, int dst, TCGv_i64 src)
{
#if defined(TARGET_RISCV32)
    TCGv_i64 tl_64 = tcg_temp_new_i64();   // low
    TCGv_i64 th_64 = tcg_temp_new_i64();   // high
    TCGv_i32 tl_32 = tcg_temp_new_i32();   // low
    TCGv_i32 th_32 = tcg_temp_new_i32();   // high
    tcg_gen_extract_i64(tl_64, src, 0, 32);
    tcg_gen_extract_i64(th_64, src, 32, 32);
    tcg_gen_trunc_i64_tl(tl_32, tl_64);
    tcg_gen_trunc_i64_tl(th_32, th_64);
    gen_set_gpr(ctx, dst, tl_32);
    gen_set_gpr(ctx, dst + 1, th_32);
# else
    gen_set_gpr(ctx, dst, src);
#endif
}

#define GEN_SIMD_TRANS(NAME)                                \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_ZPN(ctx);                                       \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1, src2);           \
    return true;                                            \
}

#define GEN_SIMD_TRANS_64(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    REQUIRE_ZPN(ctx);                                       \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1, src2);           \
    return true;                                            \
}

#define GEN_SIMD_TRANS_SHAMT(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_ZPN(ctx);                                       \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv shamt = tcg_constant_tl(a->shamt);                 \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1, shamt);          \
    gen_set_gpr(ctx, a->rd, dest);                          \
    return true;                                            \
}

#define GEN_SIMD_TRANS_SHAMT_64(NAME)                       \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    REQUIRE_ZPN(ctx);                                       \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv shamt = tcg_constant_tl(a->shamt);                 \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1, shamt);          \
    gen_set_gpr(ctx, a->rd, dest);                          \
    return true;                                            \
}

#define GEN_SIMD_TRANS_R1(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_ZPN(ctx);                                       \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1);                 \
    gen_set_gpr(ctx, a->rd, dest);                          \
    return true;                                            \
}

#define GEN_SIMD_TRANS_R1_64(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    REQUIRE_ZPN(ctx);                                       \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    gen_helper_##NAME(dest, tcg_env, src1);                 \
    gen_set_gpr(ctx, a->rd, dest);                          \
    return true;                                            \
}

#define GEN_SIMD_TRANS_RD(NAME)                             \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_ZPN(ctx);                                       \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    TCGv t = tcg_temp_new();                                \
    gen_helper_##NAME(t, tcg_env, src1, src2, dest);        \
    gen_set_gpr(ctx, a->rd, t);                             \
    return true;                                            \
}

#define GEN_SIMD_TRANS_RD_64(NAME)                          \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_64BIT(ctx);                                     \
    REQUIRE_ZPN(ctx);                                       \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv dest = dest_gpr(ctx, a->rd);                       \
    TCGv t = tcg_temp_new();                                \
    gen_helper_##NAME(t, tcg_env, src1, src2, dest);        \
    gen_set_gpr(ctx, a->rd, t);                             \
    return true;                                            \
}

#define GEN_SIMD_TRANS_ZPS_64(NAME)                         \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_ZPSFOPERAND(ctx);                               \
    TCGv_i64 rs1_64 = tcg_temp_new_i64();                   \
    TCGv_i64 rs2_64 = tcg_temp_new_i64();                   \
    TCGv_i64 rd_64 = tcg_temp_new_i64();                    \
    TCGv_i64 t = tcg_temp_new_i64();                        \
    get_pair_regs(ctx, rs1_64, a->rs1);                     \
    get_pair_regs(ctx, rs2_64, a->rs2);                     \
    get_pair_regs(ctx, rd_64, a->rd);                       \
    gen_helper_##NAME(t, tcg_env, rs1_64, rs2_64, rd_64);   \
    set_pair_regs(ctx, a->rd, t);                           \
    return true;                                            \
}

#define GEN_SIMD_TRANS_ZPS_32M_64(NAME)                     \
static bool trans_##NAME(DisasContext *ctx, arg_##NAME *a)  \
{                                                           \
    REQUIRE_ZPSFOPERAND(ctx);                               \
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);             \
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);             \
    TCGv_i64 rs1_64 = tcg_temp_new_i64();                   \
    TCGv_i64 rs2_64 = tcg_temp_new_i64();                   \
    TCGv_i64 rd_64 = tcg_temp_new_i64();                    \
    TCGv_i64 t = tcg_temp_new_i64();                        \
    get_pair_regs(ctx, rd_64, a->rd);                       \
    tcg_gen_extu_tl_i64(rs1_64, src1);                      \
    tcg_gen_extu_tl_i64(rs2_64, src2);                      \
    gen_helper_##NAME(t, tcg_env, rs1_64, rs2_64, rd_64);   \
    set_pair_regs(ctx, a->rd, t);                           \
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
GEN_SIMD_TRANS_SHAMT(srai16)
GEN_SIMD_TRANS(sra16_u)
GEN_SIMD_TRANS_SHAMT(srai16_u)
GEN_SIMD_TRANS(srl16)
GEN_SIMD_TRANS_SHAMT(srli16)
GEN_SIMD_TRANS(srl16_u)
GEN_SIMD_TRANS_SHAMT(srli16_u)
GEN_SIMD_TRANS(sll16)
GEN_SIMD_TRANS_SHAMT(slli16)
GEN_SIMD_TRANS(ksll16)
GEN_SIMD_TRANS_SHAMT(kslli16)
GEN_SIMD_TRANS(kslra16)
GEN_SIMD_TRANS(kslra16_u)

GEN_SIMD_TRANS(sra8)
GEN_SIMD_TRANS_SHAMT(srai8)
GEN_SIMD_TRANS(sra8_u)
GEN_SIMD_TRANS_SHAMT(srai8_u)
GEN_SIMD_TRANS(srl8)
GEN_SIMD_TRANS_SHAMT(srli8)
GEN_SIMD_TRANS(srl8_u)
GEN_SIMD_TRANS_SHAMT(srli8_u)
GEN_SIMD_TRANS(sll8)
GEN_SIMD_TRANS_SHAMT(slli8)
GEN_SIMD_TRANS(ksll8)
GEN_SIMD_TRANS_SHAMT(kslli8)
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
    TCGv dest1 = dest_gpr(ctx, a->rd);
    TCGv dest2 = dest_gpr(ctx, a->rd + 1);
    TCGv t1 = tcg_temp_new();
    TCGv t2 = tcg_temp_new();
    
    tcg_gen_sextract_tl(t1, src1, 0, 16);
    tcg_gen_sextract_tl(t2, src2, 0, 16);
    tcg_gen_mul_tl(dest1, t1, t2);
    tcg_gen_sextract_tl(t1, src1, 16, 16);
    tcg_gen_sextract_tl(t2, src2, 16, 16);
    tcg_gen_mul_tl(dest2, t1, t2);
    gen_set_gpr(ctx, a->rd, dest1);
    gen_set_gpr(ctx, a->rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_smul16_64(dest, tcg_env, src1, src2);
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
    TCGv dest1 = dest_gpr(ctx, a->rd);
    TCGv dest2 = dest_gpr(ctx, a->rd + 1);
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
    gen_helper_smulx16_64(dest, tcg_env, src1, src2);
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
    TCGv dest1 = dest_gpr(ctx, a->rd);
    TCGv dest2 = dest_gpr(ctx, a->rd + 1);
    TCGv t1 = tcg_temp_new();
    TCGv t2 = tcg_temp_new();

    tcg_gen_extract_tl(t1, src1, 0, 16);
    tcg_gen_extract_tl(t2, src2, 0, 16);
    tcg_gen_mul_tl(dest1, t1, t2);
    tcg_gen_extract_tl(t1, src1, 16, 16);
    tcg_gen_extract_tl(t2, src2, 16, 16);
    tcg_gen_mul_tl(dest2, t1, t2);
    gen_set_gpr(ctx, a->rd, dest1);
    gen_set_gpr(ctx, a->rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_umul16_64(dest, tcg_env, src1, src2);
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
    TCGv dest1 = dest_gpr(ctx, a->rd);
    TCGv dest2 = dest_gpr(ctx, a->rd + 1);
    TCGv t1 = tcg_temp_new();
    TCGv t2 = tcg_temp_new();

    tcg_gen_extract_tl(t1, src1, 0, 16);
    tcg_gen_extract_tl(t2, src2, 16, 16);
    tcg_gen_mul_tl(dest1, t1, t2);
    tcg_gen_extract_tl(t1, src1, 16, 16);
    tcg_gen_extract_tl(t2, src2, 0, 16);
    tcg_gen_mul_tl(dest2, t1, t2);
    gen_set_gpr(ctx, a->rd, dest1);
    gen_set_gpr(ctx, a->rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_umulx16_64(dest, tcg_env, src1, src2);
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
    TCGv dest1 = dest_gpr(ctx, a->rd);
    TCGv dest2 = dest_gpr(ctx, a->rd + 1);
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

    gen_set_gpr(ctx, a->rd, dest1);
    gen_set_gpr(ctx, a->rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_smul8_64(dest, tcg_env, src1, src2);
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
    TCGv dest1 = dest_gpr(ctx, a->rd);
    TCGv dest2 = dest_gpr(ctx, a->rd + 1);
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

    gen_set_gpr(ctx, a->rd, dest1);
    gen_set_gpr(ctx, a->rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_smulx8_64(dest, tcg_env, src1, src2);
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
    TCGv dest1 = dest_gpr(ctx, a->rd);
    TCGv dest2 = dest_gpr(ctx, a->rd + 1);
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

    gen_set_gpr(ctx, a->rd, dest1);
    gen_set_gpr(ctx, a->rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_umul8_64(dest, tcg_env, src1, src2);
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
    TCGv dest1 = dest_gpr(ctx, a->rd);
    TCGv dest2 = dest_gpr(ctx, a->rd + 1);
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

    gen_set_gpr(ctx, a->rd, dest1);
    gen_set_gpr(ctx, a->rd + 1, dest2);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_umulx8_64(dest, tcg_env, src1, src2);
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
GEN_SIMD_TRANS_SHAMT(sclip16)
GEN_SIMD_TRANS_SHAMT(uclip16)
GEN_SIMD_TRANS_R1(kabs16)
GEN_SIMD_TRANS_R1(clrs16)
GEN_SIMD_TRANS_R1(clz16)

GEN_SIMD_TRANS(smin8)
GEN_SIMD_TRANS(umin8)
GEN_SIMD_TRANS(smax8)
GEN_SIMD_TRANS(umax8)
GEN_SIMD_TRANS_SHAMT(sclip8)
GEN_SIMD_TRANS_SHAMT(uclip8)
GEN_SIMD_TRANS_R1(kabs8)
GEN_SIMD_TRANS_R1(clrs8)
GEN_SIMD_TRANS_R1(clz8)

GEN_SIMD_TRANS_R1(sunpkd810)
GEN_SIMD_TRANS_R1(sunpkd820)
GEN_SIMD_TRANS_R1(sunpkd830)
GEN_SIMD_TRANS_R1(sunpkd831)
GEN_SIMD_TRANS_R1(sunpkd832)
GEN_SIMD_TRANS_R1(zunpkd810)
GEN_SIMD_TRANS_R1(zunpkd820)
GEN_SIMD_TRANS_R1(zunpkd830)
GEN_SIMD_TRANS_R1(zunpkd831)
GEN_SIMD_TRANS_R1(zunpkd832)

GEN_SIMD_TRANS_64(pkbb16)
GEN_SIMD_TRANS(pkbt16)
GEN_SIMD_TRANS(pktb16)
GEN_SIMD_TRANS_64(pktt16)

GEN_SIMD_TRANS_64(smmul)
GEN_SIMD_TRANS(smmul_u)
GEN_SIMD_TRANS_RD(kmmac)
GEN_SIMD_TRANS_RD(kmmac_u)
GEN_SIMD_TRANS_RD(kmmsb)
GEN_SIMD_TRANS_RD(kmmsb_u)
GEN_SIMD_TRANS(kwmmul)
GEN_SIMD_TRANS(kwmmul_u)

GEN_SIMD_TRANS(smmwb)
GEN_SIMD_TRANS(smmwb_u)
GEN_SIMD_TRANS(smmwt)
GEN_SIMD_TRANS(smmwt_u)
GEN_SIMD_TRANS_RD(kmmawb)
GEN_SIMD_TRANS_RD(kmmawb_u)
GEN_SIMD_TRANS_RD(kmmawt)
GEN_SIMD_TRANS_RD(kmmawt_u)
GEN_SIMD_TRANS(kmmwb2)
GEN_SIMD_TRANS(kmmwb2_u)
GEN_SIMD_TRANS(kmmwt2)
GEN_SIMD_TRANS(kmmwt2_u)
GEN_SIMD_TRANS_RD(kmmawb2)
GEN_SIMD_TRANS_RD(kmmawb2_u)
GEN_SIMD_TRANS_RD(kmmawt2)
GEN_SIMD_TRANS_RD(kmmawt2_u)

GEN_SIMD_TRANS(smbb16)
GEN_SIMD_TRANS(smbt16)
GEN_SIMD_TRANS(smtt16)
GEN_SIMD_TRANS(kmda)
GEN_SIMD_TRANS(kmxda)
GEN_SIMD_TRANS(smds)
GEN_SIMD_TRANS(smdrs)
GEN_SIMD_TRANS(smxds)
GEN_SIMD_TRANS_RD(kmabb)
GEN_SIMD_TRANS_RD(kmabt)
GEN_SIMD_TRANS_RD(kmatt)
GEN_SIMD_TRANS_RD(kmada)
GEN_SIMD_TRANS_RD(kmaxda)
GEN_SIMD_TRANS_RD(kmads)
GEN_SIMD_TRANS_RD(kmadrs)
GEN_SIMD_TRANS_RD(kmaxds)
GEN_SIMD_TRANS_RD(kmsda)
GEN_SIMD_TRANS_RD(kmsxda)

static bool trans_smal(DisasContext *ctx, arg_smal *a)
{
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);

#if defined(TARGET_RISCV32)
    TCGv_i64 rs1_64 = tcg_temp_new_i64();
    TCGv_i64 rs2_64 = tcg_temp_new_i64();
    TCGv_i64 rd_64 = tcg_temp_new_i64();
    TCGv_i64 rs2_h = tcg_temp_new_i64();
    TCGv_i64 rs2_l = tcg_temp_new_i64();
    TCGv_i64 rs2_mul = tcg_temp_new_i64();
 
    tcg_gen_ext_i32_i64(rs2_64, src2);
    tcg_gen_sextract_i64(rs2_h, rs2_64, 16, 16);
    tcg_gen_sextract_i64(rs2_l, rs2_64, 0, 16);
    tcg_gen_mul_i64(rs2_mul, rs2_h, rs2_l);
    get_pair_regs(ctx, rs1_64, a->rs1);
    tcg_gen_add_i64(rd_64, rs1_64, rs2_mul);

    set_pair_regs(ctx, a->rd, rd_64);
#else
    TCGv dest = dest_gpr(ctx, a->rd);
    TCGv src1 = dest_gpr(ctx, a->rs1);
    gen_helper_smal_64(dest, tcg_env, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
#endif

    return true;
}

GEN_SIMD_TRANS_SHAMT(sclip32)
GEN_SIMD_TRANS_SHAMT(uclip32)
GEN_SIMD_TRANS_R1(clrs32)
GEN_SIMD_TRANS_R1(clz32)
GEN_SIMD_TRANS(pbsad)
GEN_SIMD_TRANS_RD(pbsada)

GEN_SIMD_TRANS_RD(smaqa)
GEN_SIMD_TRANS_RD(umaqa)
GEN_SIMD_TRANS_RD(smaqa_su)

static bool trans_add64(DisasContext *ctx, arg_add64 *a)
{
    REQUIRE_32BIT(ctx);
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv_i64 rs1_64 = tcg_temp_new_i64();
    TCGv_i64 rs2_64 = tcg_temp_new_i64();
    get_pair_regs(ctx, rs1_64, a->rs1);
    get_pair_regs(ctx, rs2_64, a->rs2);
    tcg_gen_add_i64(rs1_64, rs1_64, rs2_64);
    set_pair_regs(ctx, a->rd, rs1_64);
    return true;
}

GEN_SIMD_TRANS_ZPS_64(radd64)
GEN_SIMD_TRANS_ZPS_64(uradd64)
GEN_SIMD_TRANS_ZPS_64(kadd64)
GEN_SIMD_TRANS_ZPS_64(ukadd64)

static bool trans_sub64(DisasContext *ctx, arg_sub64 *a)
{
    REQUIRE_32BIT(ctx);
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv_i64 rs1_64 = tcg_temp_new_i64();
    TCGv_i64 rs2_64 = tcg_temp_new_i64();
    get_pair_regs(ctx, rs1_64, a->rs1);
    get_pair_regs(ctx, rs2_64, a->rs2);
    tcg_gen_sub_i64(rs1_64, rs1_64, rs2_64);
    set_pair_regs(ctx, a->rd, rs1_64);
    return true;
}

GEN_SIMD_TRANS_ZPS_64(rsub64)
GEN_SIMD_TRANS_ZPS_64(ursub64)
GEN_SIMD_TRANS_ZPS_64(ksub64)
GEN_SIMD_TRANS_ZPS_64(uksub64)

GEN_SIMD_TRANS_ZPS_32M_64(smar64)
GEN_SIMD_TRANS_ZPS_32M_64(smsr64)
GEN_SIMD_TRANS_ZPS_32M_64(umar64)
GEN_SIMD_TRANS_ZPS_32M_64(umsr64)
GEN_SIMD_TRANS_ZPS_32M_64(kmar64)
GEN_SIMD_TRANS_ZPS_32M_64(kmsr64)
GEN_SIMD_TRANS_ZPS_32M_64(ukmar64)
GEN_SIMD_TRANS_ZPS_32M_64(ukmsr64)

GEN_SIMD_TRANS_ZPS_32M_64(smalbb)
GEN_SIMD_TRANS_ZPS_32M_64(smalbt)
GEN_SIMD_TRANS_ZPS_32M_64(smaltt)
GEN_SIMD_TRANS_ZPS_32M_64(smalda)
GEN_SIMD_TRANS_ZPS_32M_64(smalxda)
GEN_SIMD_TRANS_ZPS_32M_64(smalds)
GEN_SIMD_TRANS_ZPS_32M_64(smaldrs)
GEN_SIMD_TRANS_ZPS_32M_64(smalxds)
GEN_SIMD_TRANS_ZPS_32M_64(smslda)
GEN_SIMD_TRANS_ZPS_32M_64(smslxda)

GEN_SIMD_TRANS(kaddh)
GEN_SIMD_TRANS(ksubh)
GEN_SIMD_TRANS(khmbb)
GEN_SIMD_TRANS(khmbt)
GEN_SIMD_TRANS(khmtt)
GEN_SIMD_TRANS(ukaddh)
GEN_SIMD_TRANS(uksubh)

GEN_SIMD_TRANS(kaddw)
GEN_SIMD_TRANS(ukaddw)
GEN_SIMD_TRANS(ksubw)
GEN_SIMD_TRANS(uksubw)
GEN_SIMD_TRANS(kdmbb)
GEN_SIMD_TRANS(kdmbt)
GEN_SIMD_TRANS(kdmtt)
GEN_SIMD_TRANS(kslraw)
GEN_SIMD_TRANS(kslraw_u)
GEN_SIMD_TRANS(ksllw)
GEN_SIMD_TRANS_SHAMT(kslliw)
GEN_SIMD_TRANS_RD(kdmabb)
GEN_SIMD_TRANS_RD(kdmabt)
GEN_SIMD_TRANS_RD(kdmatt)
GEN_SIMD_TRANS_R1(kabsw)

GEN_SIMD_TRANS(raddw)
GEN_SIMD_TRANS(uraddw)
GEN_SIMD_TRANS(rsubw)
GEN_SIMD_TRANS(ursubw)
GEN_SIMD_TRANS(maxw)
GEN_SIMD_TRANS(minw)

static bool trans_mulr64(DisasContext *ctx, arg_mulr64 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv_i64 rs1_64  = tcg_temp_new_i64();
    TCGv_i64 rs2_64  = tcg_temp_new_i64();
    TCGv_i64 rd_64 = tcg_temp_new_i64();
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    tcg_gen_ext_tl_i64(rs1_64, src1);
    tcg_gen_ext_tl_i64(rs2_64, src2);
    tcg_gen_extract_i64(rs1_64, rs1_64, 0, 32);
    tcg_gen_extract_i64(rs2_64, rs2_64, 0, 32);
    tcg_gen_mul_i64(rd_64, rs1_64, rs2_64);
    set_pair_regs(ctx, a->rd, rd_64);
    return true;
}

static bool trans_mulsr64(DisasContext *ctx, arg_mulsr64 *a)
{
    REQUIRE_ZPN(ctx);
    TCGv_i64 rs1_64  = tcg_temp_new_i64();
    TCGv_i64 rs2_64  = tcg_temp_new_i64();
    TCGv_i64 rd_64 = tcg_temp_new_i64();
    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    tcg_gen_ext_tl_i64(rs1_64, src1);
    tcg_gen_ext_tl_i64(rs2_64, src2);
    tcg_gen_sextract_i64(rs1_64, rs1_64, 0, 32);
    tcg_gen_sextract_i64(rs2_64, rs2_64, 0, 32);
    tcg_gen_mul_i64(rd_64, rs1_64, rs2_64);
    set_pair_regs(ctx, a->rd, rd_64);
    return true;
}

GEN_SIMD_TRANS(ave)
GEN_SIMD_TRANS(sra_u)
GEN_SIMD_TRANS_SHAMT(srai_u)
GEN_SIMD_TRANS(bitrev)
GEN_SIMD_TRANS_SHAMT(bitrevi)

static bool trans_wext(DisasContext *ctx, arg_wext *a)
{
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv_i64 rs1_64 = tcg_temp_new_i64();
    TCGv rs2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    get_pair_regs(ctx, rs1_64, a->rs1);
    tcg_gen_extract_tl(rs2, rs2, 0, 5);
    gen_helper_wext(dest, tcg_env, rs1_64, rs2);
    return true;
}

static bool trans_wexti(DisasContext *ctx, arg_wexti *a)
{
    REQUIRE_64BIT(ctx);
    REQUIRE_ZPSFOPERAND(ctx);
    TCGv_i64 rs1_64 = tcg_temp_new_i64();
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    get_pair_regs(ctx, rs1_64, a->rs1);
    gen_helper_wext(dest, tcg_env, rs1_64, shamt);
    return true;
}

static bool trans_cmix(DisasContext *ctx, arg_cmix *a)
{
    REQUIRE_ZBPBO(ctx);
    TCGv rs1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv rs2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv rs3 = get_gpr(ctx, a->rs3, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_cmix(dest, tcg_env, rs1, rs2, rs3);
    return true;
}

static bool trans_insb(DisasContext *ctx, arg_insb *a) {
    REQUIRE_ZPN(ctx);
    TCGv rs1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv shamt = tcg_constant_tl(a->shamt);
    TCGv dest = dest_gpr(ctx, a->rd);
    gen_helper_insb(dest, tcg_env, rs1, shamt, dest);
    return true;
}

GEN_SIMD_TRANS_RD(maddr32)
GEN_SIMD_TRANS_RD(msubr32)

GEN_SIMD_TRANS_64(add32)
GEN_SIMD_TRANS_64(radd32)
GEN_SIMD_TRANS_64(uradd32)
GEN_SIMD_TRANS_64(kadd32)
GEN_SIMD_TRANS_64(ukadd32)
GEN_SIMD_TRANS_64(sub32)
GEN_SIMD_TRANS_64(rsub32)
GEN_SIMD_TRANS_64(ursub32)
GEN_SIMD_TRANS_64(ksub32)
GEN_SIMD_TRANS_64(uksub32)
GEN_SIMD_TRANS_64(cras32)
GEN_SIMD_TRANS_64(rcras32)
GEN_SIMD_TRANS_64(urcras32)
GEN_SIMD_TRANS_64(kcras32)
GEN_SIMD_TRANS_64(ukcras32)
GEN_SIMD_TRANS_64(crsa32)
GEN_SIMD_TRANS_64(rcrsa32)
GEN_SIMD_TRANS_64(urcrsa32)
GEN_SIMD_TRANS_64(kcrsa32)
GEN_SIMD_TRANS_64(ukcrsa32)
GEN_SIMD_TRANS_64(stas32)
GEN_SIMD_TRANS_64(rstas32)
GEN_SIMD_TRANS_64(urstas32)
GEN_SIMD_TRANS_64(kstas32)
GEN_SIMD_TRANS_64(ukstas32)
GEN_SIMD_TRANS_64(stsa32)
GEN_SIMD_TRANS_64(urstsa32)
GEN_SIMD_TRANS_64(kstsa32)
GEN_SIMD_TRANS_64(ukstsa32)

GEN_SIMD_TRANS_64(sra32)
GEN_SIMD_TRANS_SHAMT_64(srai32)
GEN_SIMD_TRANS_64(sra32_u)
GEN_SIMD_TRANS_SHAMT_64(srai32_u)
GEN_SIMD_TRANS_64(srl32)
GEN_SIMD_TRANS_SHAMT_64(srli32)
GEN_SIMD_TRANS_64(srl32_u)
GEN_SIMD_TRANS_64(sll32)
GEN_SIMD_TRANS_SHAMT_64(slli32)
GEN_SIMD_TRANS_64(ksll32)
GEN_SIMD_TRANS_SHAMT_64(kslli32)
GEN_SIMD_TRANS_64(kslra32)
GEN_SIMD_TRANS_64(kslra32_u)

GEN_SIMD_TRANS_64(smin32)
GEN_SIMD_TRANS_64(umin32)
GEN_SIMD_TRANS_64(smax32)
GEN_SIMD_TRANS_R1_64(kabs32)

GEN_SIMD_TRANS_64(khmbb16)
GEN_SIMD_TRANS_64(khmbt16)
GEN_SIMD_TRANS_64(khmtt16)
GEN_SIMD_TRANS_64(kdmbb16)
GEN_SIMD_TRANS_64(kdmbt16)
GEN_SIMD_TRANS_64(kdmtt16)
GEN_SIMD_TRANS_RD_64(kdmabb16)
GEN_SIMD_TRANS_RD_64(kdmabt16)
GEN_SIMD_TRANS_RD_64(kdmatt16)

GEN_SIMD_TRANS_64(smbt32)
GEN_SIMD_TRANS_64(smtt32)

GEN_SIMD_TRANS_RD_64(kmabb32)
GEN_SIMD_TRANS_RD_64(kmabt32)
GEN_SIMD_TRANS_RD_64(kmatt32)

GEN_SIMD_TRANS_64(kmda32)
GEN_SIMD_TRANS_64(kmxda32)
GEN_SIMD_TRANS_RD_64(kmaxda32)
GEN_SIMD_TRANS_RD_64(kmads32)
GEN_SIMD_TRANS_RD_64(kmadrs32)
GEN_SIMD_TRANS_RD_64(kmaxds32)
GEN_SIMD_TRANS_RD_64(kmsda32)
GEN_SIMD_TRANS_RD_64(kmsxda32)
GEN_SIMD_TRANS_64(smds32)
GEN_SIMD_TRANS_64(smdrs32)
GEN_SIMD_TRANS_64(smxds32)
