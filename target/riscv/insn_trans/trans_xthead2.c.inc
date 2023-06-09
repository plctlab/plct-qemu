/*
 * RISC-V translation routines for the T-Head Extend Instruction Set.
 *
 * Copyright (c) 2020 T-Head Limited, ltd
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

static bool trans_ipush(DisasContext *ctx, arg_ipush *a)
{
#ifndef CONFIG_USER_ONLY
    gen_helper_ipush(cpu_env);
    return true;
#else
    return false;
#endif
}

static bool trans_ipop(DisasContext *ctx, arg_ipop *a)
{
#ifndef CONFIG_USER_ONLY
    tcg_gen_movi_tl(cpu_pc, ctx->base.pc_next);
    gen_helper_ipop(cpu_pc, cpu_env, cpu_pc);
    exit_tb(ctx); /* no chaining */
    ctx->base.is_jmp = DISAS_NORETURN;
    return true;
#else
    return false;
#endif
}

static bool trans_ignore(DisasContext *ctx, arg_ignore *a)
{
    tcg_gen_movi_tl(cpu_pc, ctx->pc_succ_insn);
    exit_tb(ctx);
    ctx->base.is_jmp = DISAS_NORETURN;
    return true;
}
