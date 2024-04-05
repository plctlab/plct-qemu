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
