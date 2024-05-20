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

target_ulong HELPER(sub16)(target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)(v1 - v2);
    }

    return rd;
}

target_ulong HELPER(rsub16)(target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)(v1 - v2) >> 1;
    }

    return rd;
}

target_ulong HELPER(ursub16)(target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)(v1 - v2) >> 1;
    }

    return rd;
}

target_ulong HELPER(ksub16)(target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)signed_saturate(v1 - v2, 16);
    }

    return rd;
}

target_ulong HELPER(uksub16)(target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)unsigned_saturate(v1 - v2, 16);
    }

    return rd;
}

target_ulong HELPER(cras16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)(v1 + v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)(v1 - v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(rcras16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)(v1 + v2) >> 1;
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)(v1 - v2) >> 1;
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(urcras16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (uint16_t)(v1 + v2) >> 1;
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint16_t)(v1 - v2) >> 1;
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kcras16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)signed_saturate(v1 + v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)signed_saturate(v1 - v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukcras16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (uint16_t)unsigned_saturate(v1 + v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint16_t)unsigned_saturate(v1 - v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(crsa16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)(v1 - v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)(v1 + v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(rcrsa16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)(v1 - v2) >> 1;
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)(v1 + v2) >> 1;
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(urcrsa16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (uint16_t)(v1 - v2) >> 1;
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint16_t)(v1 + v2) >> 1;
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kcrsa16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int16_t)signed_saturate(v1 - v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)signed_saturate(v1 + v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukcrsa16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (uint16_t)unsigned_saturate(v1 - v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint16_t)unsigned_saturate(v1 + v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(stas16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(v1 + v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int16_t)(v1 - v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(rstas16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(v1 + v2) >> 1;
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int16_t)(v1 - v2) >> 1;
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(urstas16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)(v1 + v2) >> 1;
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint16_t)(v1 - v2) >> 1;
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kstas16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)signed_saturate(v1 + v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int16_t)signed_saturate(v1 - v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukstas16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)unsigned_saturate(v1 + v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint16_t)unsigned_saturate(v1 - v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(stsa16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(v1 - v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int16_t)(v1 + v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(rstsa16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)(v1 - v2) >> 1;
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int16_t)(v1 + v2) >> 1;
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(urstsa16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)(v1 - v2) >> 1;
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint16_t)(v1 + v2) >> 1;
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kstsa16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int16_t)signed_saturate(v1 - v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int16_t)signed_saturate(v1 + v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukstsa16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 2) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint16_t)unsigned_saturate(v1 - v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint16_t)unsigned_saturate(v1 + v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(add8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(radd8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)(v1 + v2) >> 1;
    }

    return rd;
}

target_ulong HELPER(uradd8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)(v1 + v2) >> 1;
    }

    return rd;
}

target_ulong HELPER(kadd8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)signed_saturate(v1 + v2, 8);
    }

    return rd;
}

target_ulong HELPER(ukadd8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)unsigned_saturate(v1 + v2, 8);
    }

    return rd;
}

target_ulong HELPER(sub8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)(v1 - v2);
    }

    return rd;
}

target_ulong HELPER(rsub8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)(v1 - v2) >> 1;
    }

    return rd;
}

target_ulong HELPER(ursub8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)(v1 - v2) >> 1;
    }

    return rd;
}

target_ulong HELPER(ksub8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int8_t)signed_saturate(v1 - v2, 8);
    }

    return rd;
}

target_ulong HELPER(uksub8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint8_t)unsigned_saturate(v1 - v2, 8);
    }

    return rd;
}

target_ulong HELPER(sra16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_ulong shamt = rs2 & 0xF;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(srai16)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(sra16_u)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_ulong shamt = rs2 & 0xF;
    target_long v1 = 0;

    if(shamt == 0) {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            rd_p[i] = rs1_p[i];
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int16_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srai16_u)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;

    if(shamt == 0) {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            rd_p[i] = rs1_p[i];
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int16_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srl16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong shamt = rs2 & 0xF;
    target_ulong v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint16_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(srli16)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint16_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(srl16_u)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong shamt = rs2 & 0xF;
    target_ulong v1 = 0;

    if(shamt == 0) {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            rd_p[i] = rs1_p[i];
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (uint16_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srli16_u)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;

    if(shamt == 0) {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            rd_p[i] = rs1_p[i];
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (uint16_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(sll16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_ulong shamt = rs2 & 0xF;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 << shamt);
    }

    return rd;
}

target_ulong HELPER(slli16)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)(v1 << shamt);
    }

    return rd;
}

target_ulong HELPER(ksll16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_ulong shamt = rs2 & 0xF;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)signed_saturate(v1 << shamt, 16);
    }

    return rd;
}

target_ulong HELPER(kslli16)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)signed_saturate(v1 << shamt, 16);
    }

    return rd;
}

target_ulong HELPER(kslra16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_ulong shamt = rs2 & 0x1F;
    target_long v1 = 0;

    if(shamt & 0x10) {
        shamt = 16 - (shamt & 0xF);
        if(shamt == 16)
            shamt = 15;

        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int16_t)(v1 >> shamt);
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int16_t)signed_saturate(v1 << shamt, 16);
        }
    }

    return rd;
}

target_ulong HELPER(kslra16_u)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_ulong shamt = rs2 & 0x1F;
    target_long v1 = 0;

    if(shamt & 0x10) {
        shamt = 16 - (shamt & 0xF);
        if(shamt == 16)
            shamt = 15;

        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int16_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int16_t)signed_saturate(v1 << shamt, 16);
        }
    }

    return rd;
}

target_ulong HELPER(sra8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_ulong shamt = rs2 & 0x7;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(srai8)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(sra8_u)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_ulong shamt = rs2 & 0x7;
    target_long v1 = 0;

    if(shamt == 0) {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            rd_p[i] = rs1_p[i];
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int8_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srai8_u)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;

    if(shamt == 0) {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            rd_p[i] = rs1_p[i];
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int8_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srl8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong shamt = rs2 & 0x7;
    target_ulong v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint8_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(srli8)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint8_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(srl8_u)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong shamt = rs2 & 0x7;
    target_ulong v1 = 0;

    if(shamt == 0) {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            rd_p[i] = rs1_p[i];
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (uint8_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srli8_u)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong v1 = 0;

    if(shamt == 0) {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            rd_p[i] = rs1_p[i];
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (uint8_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(sll8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_ulong shamt = rs2 & 0x7;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 << shamt);
    }

    return rd;
}

target_ulong HELPER(slli8)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)(v1 << shamt);
    }

    return rd;
}

target_ulong HELPER(ksll8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_ulong shamt = rs2 & 0x7;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)signed_saturate(v1 << shamt, 8);
    }

    return rd;
}

target_ulong HELPER(kslli8)(target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)signed_saturate(v1 << shamt, 8);
    }

    return rd;
}

target_ulong HELPER(kslra8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_ulong shamt = rs2 & 0xF;
    target_long v1 = 0;

    if(shamt & 0x8) {
        shamt = 8 - (shamt & 0x7);
        if(shamt == 8)
            shamt = 7;

        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int8_t)(v1 >> shamt);
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int8_t)signed_saturate(v1 << shamt, 8);
        }
    }

    return rd;
}

target_ulong HELPER(kslra8_u)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_ulong shamt = rs2 & 0xF;
    target_long v1 = 0;

    if(shamt & 0x8) {
        shamt = 8 - (shamt & 0x7);
        if(shamt == 8)
            shamt = 7;

        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int8_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int8_t)signed_saturate(v1 << shamt, 8);
        }
    }

    return rd;
}

target_ulong HELPER(cmpeq16)(target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 == v2 ? 0xFFFF : 0;
    }

    return rd;
}

target_ulong HELPER(scmplt16)(target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 < v2 ? 0xFFFF : 0;
    }

    return rd;
}

target_ulong HELPER(scmple16)(target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 <= v2 ? 0xFFFF : 0;
    }

    return rd;
}

target_ulong HELPER(ucmplt16)(target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 < v2 ? 0xFFFF : 0;
    }

    return rd;
}

target_ulong HELPER(ucmple16)(target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 <= v2 ? 0xFFFF : 0;
    }

    return rd;
}

target_ulong HELPER(cmpeq8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 == v2 ? 0xFF : 0;
    }

    return rd;
}

target_ulong HELPER(scmplt8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? 0xFF : 0;
    }

    return rd;
}

target_ulong HELPER(scmple8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 <= v2 ? 0xFF : 0;
    }

    return rd;
}

target_ulong HELPER(ucmplt8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 < v2 ? 0xFF : 0;
    }

    return rd;
}

target_ulong HELPER(ucmple8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 <= v2 ? 0xFF : 0;
    }

    return rd;
}

target_ulong HELPER(smul16_64)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 * v2;
    }

    return rd;
}

target_ulong HELPER(smulx16_64)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[1 - i];
        rd_p[i] = v1 * v2;
    }

    return rd;
}

target_ulong HELPER(umul16_64)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 * v2;
    }

    return rd;
}

target_ulong HELPER(umulx16_64)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < 2; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[1 - i];
        rd_p[i] = v1 * v2;
    }

    return rd;
}

target_ulong HELPER(khm16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int16_t op1t, op1b, op2t, op2b;
    int16_t rest, resb;
    int i = 0;

    while(i < TARGET_LONG_SIZE / 2) {
        op1t = rs1_p[i + 1];
        op2t = rs2_p[i + 1];
        op1b = rs1_p[i];
        op2b = rs2_p[i];

        if(op1t != INT16_MIN || op2t != INT16_MIN) {
            rest = (op1t * op2t) >> 15;
        } else {
            rest = INT16_MAX;
        }

        if(op1b != INT16_MIN || op2b != INT16_MIN) {
            resb = (op1b * op2b) >> 15;
        } else {
            resb = INT16_MAX;
        }

        rd_p[i / 2] = (rest << 16) | (resb & 0xFFFF);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(khmx16)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int16_t op1t, op1b, op2t, op2b;
    int16_t rest, resb;
    int i = 0;

    while(i < TARGET_LONG_SIZE / 2) {
        op1t = rs1_p[i + 1];
        op2t = rs2_p[i];
        op1b = rs1_p[i];
        op2b = rs2_p[i + 1];

        if(op1t != INT16_MIN || op2t != INT16_MIN) {
            rest = (op1t * op2t) >> 15;
        } else {
            rest = INT16_MAX;
        }

        if(op1b != INT16_MIN || op2b != INT16_MIN) {
            resb = (op1b * op2b) >> 15;
        } else {
            resb = INT16_MAX;
        }

        rd_p[i / 2] = (rest << 16) | (resb & 0xFFFF);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(smul8_64)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 * v2;
    }

    return rd;
}

target_ulong HELPER(smulx8_64)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < 4) {
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = v1 * v2;
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = v1 * v2;
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(umul8_64)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = v1 * v2;
    }

    return rd;
}

target_ulong HELPER(umulx8_64)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rs2_p = (uint8_t*)&rs2;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < 4) {
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = v1 * v2;
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = v1 * v2;
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(khm8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    int8_t op1t, op1b, op2t, op2b;
    int8_t rest, resb;
    int i = 0;

    while(i < TARGET_LONG_SIZE) {
        op1t = rs1_p[i + 1];
        op2t = rs2_p[i + 1];
        op1b = rs1_p[i];
        op2b = rs2_p[i];

        if(op1t != INT8_MIN || op2t != INT8_MIN) {
            rest = (op1t * op2t) >> 7;
        } else {
            rest = INT8_MAX;
        }

        if(op1b != INT8_MIN || op2b != INT8_MIN) {
            resb = (op1b * op2b) >> 7;
        } else {
            resb = INT8_MAX;
        }

        rd_p[i / 2] = (rest << 8) | (resb & 0xFF);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(khmx8)(target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    int16_t *rd_p = (int16_t*)&rd;
    int8_t op1t, op1b, op2t, op2b;
    int8_t rest, resb;
    int i = 0;

    while(i < TARGET_LONG_SIZE) {
        op1t = rs1_p[i + 1];
        op2t = rs2_p[i];
        op1b = rs1_p[i];
        op2b = rs2_p[i + 1];

        if(op1t != INT8_MIN || op2t != INT8_MIN) {
            rest = (op1t * op2t) >> 7;
        } else {
            rest = INT8_MAX;
        }

        if(op1b != INT8_MIN || op2b != INT8_MIN) {
            resb = (op1b * op2b) >> 7;
        } else {
            resb = INT8_MAX;
        }

        rd_p[i / 2] = (rest << 8) | (resb & 0xFF);
        i = i + 2;
    }

    return rd;
}
