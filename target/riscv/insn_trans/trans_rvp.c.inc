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

static bool trans_add16(DisasContext *ctx, arg_add16 *a)
{
    REQUIRE_ZPN(ctx);

    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);

    gen_helper_add16(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_radd16(DisasContext *ctx, arg_radd16 *a)
{
    REQUIRE_ZPN(ctx);

    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);

    gen_helper_radd16(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_uradd16(DisasContext *ctx, arg_uradd16 *a)
{
    REQUIRE_ZPN(ctx);

    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);

    gen_helper_uradd16(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_kadd16(DisasContext *ctx, arg_kadd16 *a)
{
    REQUIRE_ZPN(ctx);

    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);

    gen_helper_kadd16(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}

static bool trans_ukadd16(DisasContext *ctx, arg_ukadd16 *a)
{
    REQUIRE_ZPN(ctx);

    TCGv src1 = get_gpr(ctx, a->rs1, EXT_NONE);
    TCGv src2 = get_gpr(ctx, a->rs2, EXT_NONE);
    TCGv dest = dest_gpr(ctx, a->rd);

    gen_helper_ukadd16(dest, src1, src2);
    gen_set_gpr(ctx, a->rd, dest);
    return true;
}
