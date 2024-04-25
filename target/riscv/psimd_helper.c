/*
 * RISC-V Packed SIMD Helpers for QEMU.
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

#include "qemu/osdep.h"
#include "cpu.h"
#include "qemu/host-utils.h"
#include "exec/exec-all.h"
#include "exec/helper-proto.h"
#include "fpu/softfloat.h"
#include "internals.h"

static target_long signed_saturate(target_long arg, int range)
{
    target_long max = (1 << (range - 1)) - 1;
    target_long min = - (1 << (range - 1));
    
    if(arg > max)
        arg = max;
    else if(arg < min)
        arg = min;
    
    return arg;
}

static target_ulong unsigned_saturate(target_ulong arg, int range)
{
    target_ulong max = (1 << range) - 1;
    
    if(arg > max)
        arg = max;
    
    return arg;
}

target_ulong HELPER(add16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(radd16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(v1 + v2) >> 1;
    }

    return rd;
}

target_ulong HELPER(uradd16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)(v1 + v2) >> 1;
    }

    return rd;
}

target_ulong HELPER(kadd16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)signed_saturate(v1 + v2, 16);
    }

    return rd;
}

target_ulong HELPER(ukadd16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)unsigned_saturate(v1 + v2, 16);
    }

    return rd;
}
