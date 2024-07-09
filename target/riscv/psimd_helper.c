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

static int64_t signed_saturate(CPURISCVState *env, int64_t arg, int range)
{
    int64_t max = (1ull << (range - 1)) - 1;
    int64_t min = - (1 << (range - 1));
    
    if(arg > max) {
        arg = max;
        env->vxsat = 0x1;
    }
    else if(arg < min) {
        arg = min;
        env->vxsat = 0x1;
    }
    
    return arg;
}

static int64_t signed_saturate64(CPURISCVState *env, Int128 arg)
{
    Int128 max = INT64_MAX;
    Int128 min = INT64_MIN;

    if(arg > max) {
        arg = max;
        env->vxsat = 0x1;
    }
    else if(arg < min) {
        arg = min;
        env->vxsat = 0x1;
    }

    return arg;
}

static uint64_t unsigned_saturate(CPURISCVState *env, uint64_t arg, int range)
{
    uint64_t max = (1ull << range) - 1;
    
    if(arg > max) {
        arg = max;
        env->vxsat = 0x1;
    }
    
    return arg;
}

static uint64_t unsigned_saturate64(CPURISCVState *env, __uint128_t arg)
{
    __uint128_t max = UINT64_MAX;
    
    if(arg > max) {
        arg = max;
        env->vxsat = 0x1;
    }
    
    return arg;
}

target_ulong HELPER(add16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(radd16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(uradd16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(kadd16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)signed_saturate(env, v1 + v2, 16);
    }

    return rd;
}

target_ulong HELPER(ukadd16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)unsigned_saturate(env, v1 + v2, 16);
    }

    return rd;
}

target_ulong HELPER(sub16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(rsub16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(ursub16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(ksub16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)signed_saturate(env, v1 - v2, 16);
    }

    return rd;
}

target_ulong HELPER(uksub16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)unsigned_saturate(env, v1 - v2, 16);
    }

    return rd;
}

target_ulong HELPER(cras16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(rcras16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)((v1 + v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(urcras16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)((v1 + v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint16_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kcras16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)signed_saturate(env, v1 + v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)signed_saturate(env, v1 - v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukcras16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)unsigned_saturate(env, v1 + v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint16_t)unsigned_saturate(env, v1 - v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(crsa16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(rcrsa16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)((v1 - v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)((v1 + v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(urcrsa16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)((v1 - v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint16_t)((v1 + v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kcrsa16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)signed_saturate(env, v1 - v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int16_t)signed_saturate(env, v1 + v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukcrsa16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)unsigned_saturate(env, v1 - v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint16_t)unsigned_saturate(env, v1 + v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(stas16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(rstas16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)((v1 + v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int16_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(urstas16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)((v1 + v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint16_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kstas16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)signed_saturate(env, v1 + v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int16_t)signed_saturate(env, v1 - v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukstas16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)unsigned_saturate(env, v1 + v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint16_t)unsigned_saturate(env, v1 - v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(stsa16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(urstsa16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)((v1 - v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint16_t)((v1 + v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kstsa16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int16_t)signed_saturate(env, v1 - v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int16_t)signed_saturate(env, v1 + v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukstsa16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint16_t)unsigned_saturate(env, v1 - v2, 16);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint16_t)unsigned_saturate(env, v1 + v2, 16);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(add8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(radd8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int8_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(uradd8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint8_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(kadd8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int8_t)signed_saturate(env, v1 + v2, 8);
    }

    return rd;
}

target_ulong HELPER(ukadd8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint8_t)unsigned_saturate(env, v1 + v2, 8);
    }

    return rd;
}

target_ulong HELPER(sub8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(rsub8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int8_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(ursub8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint8_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(ksub8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (int8_t)signed_saturate(env, v1 - v2, 8);
    }

    return rd;
}

target_ulong HELPER(uksub8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = (uint8_t)unsigned_saturate(env, v1 - v2, 8);
    }

    return rd;
}

target_ulong HELPER(sra16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(srai16)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
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

target_ulong HELPER(sra16_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_ulong shamt = rs2 & 0xF;
    target_long v1 = 0;

    if(shamt == 0) {
        rd = rs1;
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int16_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srai16_u)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;

    if(shamt == 0) {
        rd = rs1;
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int16_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srl16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(srli16)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
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

target_ulong HELPER(srl16_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong shamt = rs2 & 0xF;
    target_ulong v1 = 0;

    if(shamt == 0) {
        rd = rs1;
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (uint16_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srli16_u)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;

    if(shamt == 0) {
        rd = rs1;
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (uint16_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(sll16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(slli16)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
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

target_ulong HELPER(ksll16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_ulong shamt = rs2 & 0xF;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)signed_saturate(env, v1 << shamt, 16);
    }

    return rd;
}

target_ulong HELPER(kslli16)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)signed_saturate(env, v1 << shamt, 16);
    }

    return rd;
}

target_ulong HELPER(kslra16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
            rd_p[i] = (int16_t)signed_saturate(env, v1 << shamt, 16);
        }
    }

    return rd;
}

target_ulong HELPER(kslra16_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
            rd_p[i] = (int16_t)signed_saturate(env, v1 << shamt, 16);
        }
    }

    return rd;
}

target_ulong HELPER(sra8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(srai8)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
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

target_ulong HELPER(sra8_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_ulong shamt = rs2 & 0x7;
    target_long v1 = 0;

    if(shamt == 0) {
        rd = rs1;
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int8_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srai8_u)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;

    if(shamt == 0) {
        rd = rs1;
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int8_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srl8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(srli8)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
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

target_ulong HELPER(srl8_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong shamt = rs2 & 0x7;
    target_ulong v1 = 0;

    if(shamt == 0) {
        rd = rs1;
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (uint8_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srli8_u)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong v1 = 0;

    if(shamt == 0) {
        rd = rs1;
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (uint8_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(sll8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(slli8)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
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

target_ulong HELPER(ksll8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_ulong shamt = rs2 & 0x7;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)signed_saturate(env, v1 << shamt, 8);
    }

    return rd;
}

target_ulong HELPER(kslli8)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)signed_saturate(env, v1 << shamt, 8);
    }

    return rd;
}

target_ulong HELPER(kslra8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
            rd_p[i] = (int8_t)signed_saturate(env, v1 << shamt, 8);
        }
    }

    return rd;
}

target_ulong HELPER(kslra8_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
            rd_p[i] = (int8_t)signed_saturate(env, v1 << shamt, 8);
        }
    }

    return rd;
}

target_ulong HELPER(cmpeq16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(scmplt16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(scmple16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(ucmplt16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(ucmple16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(cmpeq8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(scmplt8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(scmple8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(ucmplt8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(ucmple8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(smul16_64)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(smulx16_64)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(umul16_64)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(umulx16_64)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(khm16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
            env->vxsat = 0x1;
        }

        if(op1b != INT16_MIN || op2b != INT16_MIN) {
            resb = (op1b * op2b) >> 15;
        } else {
            resb = INT16_MAX;
            env->vxsat = 0x1;
        }

        rd_p[i / 2] = (rest << 16) | (resb & 0xFFFF);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(khmx16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
            env->vxsat = 0x1;
        }

        if(op1b != INT16_MIN || op2b != INT16_MIN) {
            resb = (op1b * op2b) >> 15;
        } else {
            resb = INT16_MAX;
            env->vxsat = 0x1;
        }

        rd_p[i / 2] = (rest << 16) | (resb & 0xFFFF);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(smul8_64)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(smulx8_64)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(umul8_64)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(umulx8_64)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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

target_ulong HELPER(khm8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
            env->vxsat = 0x1;
        }

        if(op1b != INT8_MIN || op2b != INT8_MIN) {
            resb = (op1b * op2b) >> 7;
        } else {
            resb = INT8_MAX;
            env->vxsat = 0x1;
        }

        rd_p[i / 2] = (rest << 8) | (resb & 0xFF);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(khmx8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
            env->vxsat = 0x1;
        }

        if(op1b != INT8_MIN || op2b != INT8_MIN) {
            resb = (op1b * op2b) >> 7;
        } else {
            resb = INT8_MAX;
            env->vxsat = 0x1;
        }

        rd_p[i / 2] = (rest << 8) | (resb & 0xFF);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(smin16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(umin16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(smax16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(umax16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(sclip16)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int16_t)signed_saturate(env, v1, shamt + 1);
    }

    return rd;
}

target_ulong HELPER(uclip16)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        target_long max = (1 << shamt) - 1;
        if(v1 > max) {
            v1 = max;
            env->vxsat = 0x1;
        } else if(v1 < 0) {
            v1 = 0;
            env->vxsat = 0x1;
        }
        rd_p[i] = v1;
    }

    return rd;
}

target_ulong HELPER(kabs16)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rd_p = (int16_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];

        if(v1 == INT16_MIN) {
            v1 = INT16_MAX;
            env->vxsat = 0x1;
        } else if(v1 < 0) {
            v1 = -v1;
        }

        rd_p[i] = v1;
    }

    return rd;
}

target_ulong HELPER(clrs16)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        v1 = rs1_p[i];
        int sign = v1 >> 15;
        int cnt = 0;
        
        for(int j = 14; j >=0; j--) {
            if(((v1 >> j) & 1) == sign) {
                cnt++;
            } else {
                break;
            }
        }

        rd_p[i] = cnt;
    }

    return rd;
}

target_ulong HELPER(clz16)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rd_p = (uint16_t*)&rd;
    target_ulong v1 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 2; i++) {
        int cnt = 0;
        v1 = rs1_p[i];
        
        for(int j = 15; j >=0; j--) {
            if((v1 >> j) == 0) {
                cnt++;
            } else {
                break;
            }
        }    

        rd_p[i] = cnt;
    }

    return rd;
}

target_ulong HELPER(smin8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(umin8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 < v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(smax8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(umax8)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
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
        rd_p[i] = v1 > v2 ? v1 : v2;
    }

    return rd;
}

target_ulong HELPER(sclip8)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int8_t)signed_saturate(env, v1, shamt + 1);
    }

    return rd;
}

target_ulong HELPER(uclip8)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        target_long max = (1 << shamt) - 1;
        if(v1 > max) {
            v1 = max;
            env->vxsat = 0x1;
        } else if(v1 < 0) {
            v1 = 0;
            env->vxsat = 0x1;
        }
        rd_p[i] = v1;
    }

    return rd;
}

target_ulong HELPER(kabs8)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rd_p = (int8_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];

        if(v1 == INT8_MIN) {
            v1 = INT8_MAX;
            env->vxsat = 0x1;
        } else if(v1 < 0) {
            v1 = -v1;
        }

        rd_p[i] = v1;
    }

    return rd;
}

target_ulong HELPER(clrs8)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        int sign = v1 >> 7;
        int cnt = 0;
        
        for(int j = 6; j >=0; j--) {
            if(((v1 >> j) & 1) == sign) {
                cnt++;
            } else {
                break;
            }
        }

        rd_p[i] = cnt;
    }

    return rd;
}

target_ulong HELPER(clz8)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rd_p = (uint8_t*)&rd;
    target_ulong v1 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        int cnt = 0;
        v1 = rs1_p[i];
        
        for(int j = 7; j >=0; j--) {
            if((v1 >> j) == 0) {
                cnt++;
            } else {
                break;
            }
        }    

        rd_p[i] = cnt;
    }

    return rd;
}

target_ulong HELPER(sunpkd810)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = sextract32(v1, 8, 8);
        t2 = sextract32(v1, 0, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(sunpkd820)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = sextract32(v1, 16, 8);
        t2 = sextract32(v1, 0, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(sunpkd830)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = sextract32(v1, 24, 8);
        t2 = sextract32(v1, 0, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(sunpkd831)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = sextract32(v1, 24, 8);
        t2 = sextract32(v1, 8, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(sunpkd832)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = sextract32(v1, 24, 8);
        t2 = sextract32(v1, 16, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(zunpkd810)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = extract32(v1, 8, 8);
        t2 = extract32(v1, 0, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(zunpkd820)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = extract32(v1, 16, 8);
        t2 = extract32(v1, 0, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(zunpkd830)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = extract32(v1, 24, 8);
        t2 = extract32(v1, 0, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(zunpkd831)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = extract32(v1, 24, 8);
        t2 = extract32(v1, 8, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(zunpkd832)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        t1 = extract32(v1, 24, 8);
        t2 = extract32(v1, 16, 8);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}


target_ulong HELPER(pkbb16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t1 = extract32(v1, 0, 16);
        t2 = extract32(v2, 0, 16);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(pkbt16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t1 = extract32(v1, 0, 16);
        t2 = extract32(v2, 16, 16);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(pktb16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t1 = extract32(v1, 16, 16);
        t2 = extract32(v2, 0, 16);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(pktt16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    uint32_t t1 = 0;
    uint32_t t2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t1 = extract32(v1, 16, 16);
        t2 = extract32(v2, 16, 16);

        rd_p[i] = (t1 << 16) | (t2 & 0xFFFF);
    }

    return rd;
}

target_ulong HELPER(smmul)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t = v1 * v2;
        rd_p[i] = (int32_t)(t >> 32);
    }

    return rd;
}

target_ulong HELPER(smmul_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t = v1 * v2;
        rd_p[i] = (int32_t)((((uint64_t)t >> 31) + 1) >> 1);
    }

    return rd;
}

target_ulong HELPER(kmmac)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        v3 = rd_p[i];
        t = v1 * v2;
        rd_p[i] = signed_saturate(env, v3 + ((uint64_t)t >> 32), 32);
    }

    return rd;
}

target_ulong HELPER(kmmac_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t1 = 0;
    int32_t t2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        v3 = rd_p[i];
        t1 = v1 * v2;
        /* sign bits are cut and then expanded */ 
        t2 = (((uint64_t)t1 >> 31) + 1) >> 1;
        rd_p[i] = signed_saturate(env, v3 + (int64_t)t2, 32);
    }

    return rd;
}

target_ulong HELPER(kmmsb)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        v3 = rd_p[i];
        t = v1 * v2;
        rd_p[i] = signed_saturate(env, v3 - ((uint64_t)t >> 32), 32);
    }

    return rd;
}

target_ulong HELPER(kmmsb_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t1 = 0;
    int32_t t2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        v3 = rd_p[i];
        t1 = v1 * v2;
        /* sign bits are cut and then expanded */ 
        t2 = ((uint64_t)t1 >> 31) + 1;
        rd_p[i] = signed_saturate(env, v3 - (int64_t)t2, 32);
    }

    return rd;
}

target_ulong HELPER(kwmmul)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];

        if(v1 == INT32_MIN && v2 == INT32_MIN) {
            rd_p[i] = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            t = v1 * v2;
            rd_p[i] = (int32_t)(t >> 31);
        }
    }

    return rd;
}

target_ulong HELPER(kwmmul_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];

        if(v1 == INT32_MIN && v2 == INT32_MIN) {
            rd_p[i] = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            t = v1 * v2;
            rd_p[i] = (int32_t)((((uint64_t)t >> 30) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(smmwb)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        t = v1 * v2;
        rd_p[i] = (int32_t)(t >> 16);
    }

    return rd;
}

target_ulong HELPER(smmwb_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        t = v1 * v2;
        rd_p[i] = (int32_t)((((uint64_t)t >> 15) + 1) >> 1);
    }

    return rd;
}

target_ulong HELPER(smmwt)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)(rs2_p[i] >> 16);
        t = v1 * v2;
        rd_p[i] = (int32_t)(t >> 16);
    }

    return rd;
}

target_ulong HELPER(smmwt_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)(rs2_p[i] >> 16);
        t = v1 * v2;
        rd_p[i] = (int32_t)((((uint64_t)t >> 15) + 1) >> 1);
    }

    return rd;
}

target_ulong HELPER(kmmawb)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        v3 = rd_p[i];
        t = v1 * v2;
        rd_p[i] = (int32_t)signed_saturate(env, v3 + ((uint64_t)t >> 16), 32);
    }

    return rd;
}

target_ulong HELPER(kmmawb_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t1 = 0;
    int32_t t2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        v3 = rd_p[i];
        t1 = v1 * v2;
        /* sign bits are cut and then expanded */ 
        t2 = (((uint64_t)t1 >> 15) + 1) >> 1;
        rd_p[i] = (int32_t)signed_saturate(env, v3 + (int64_t)t2, 32);
    }

    return rd;
}

target_ulong HELPER(kmmawt)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)(rs2_p[i] >> 16);
        v3 = rd_p[i];
        t = v1 * v2;
        rd_p[i] = (int32_t)signed_saturate(env, v3 + ((uint64_t)t >> 16), 32);
    }

    return rd;
}

target_ulong HELPER(kmmawt_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t1 = 0;
    int32_t t2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)(rs2_p[i] >> 16);
        v3 = rd_p[i];
        t1 = v1 * v2;
        /* sign bits are cut and then expanded */ 
        t2 = (((uint64_t)t1 >> 15) + 1) >> 1;
        rd_p[i] = (int32_t)signed_saturate(env, v3 + (int64_t)t2, 32);
    }

    return rd;
}

target_ulong HELPER(kmmwb2)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)rs2_p[i];

        if(v1 == INT32_MIN && v2 == INT16_MIN) {
            rd_p[i] = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            t = v1 * v2;
            rd_p[i] = (int32_t)(t >> 15);
        }
    }

    return rd;
}

target_ulong HELPER(kmmwb2_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)rs2_p[i];

        if(v1 == INT32_MIN && v2 == INT16_MIN) {
            rd_p[i] = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            t = v1 * v2;
            rd_p[i] = (int32_t)((((uint64_t)t >> 14) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(kmmwt2)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)(rs2_p[i] >> 16);

        if(v1 == INT32_MIN && v2 == INT16_MIN) {
            rd_p[i] = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            t = v1 * v2;
            rd_p[i] = (int32_t)(t >> 15);
        }
    }

    return rd;
}

target_ulong HELPER(kmmwt2_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)(rs2_p[i] >> 16);

        if(v1 == INT32_MIN && v2 == INT16_MIN) {
            rd_p[i] = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            t = v1 * v2;
            rd_p[i] = (int32_t)((((uint64_t)t >> 14) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(kmmawb2)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t = 0;
    int64_t addop = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        v3 = rd_p[i];

        if(v1 == INT32_MIN && v2 == INT16_MIN) {
            addop = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            t = v1 * v2;
            addop = (int32_t)(t >> 15);
        }

        rd_p[i] = (int32_t)signed_saturate(env, v3 + addop, 32);
    }

    return rd;
}

target_ulong HELPER(kmmawb2_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t = 0;
    int64_t addop = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        v3 = rd_p[i];

        if(v1 == INT32_MIN && v2 == INT16_MIN) {
            addop = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            t = v1 * v2;
            addop = (int32_t)((((uint64_t)t >> 14) + 1) >> 1);
        }
        
        rd_p[i] = (int32_t)signed_saturate(env, v3 + addop, 32);
    }

    return rd;
}

target_ulong HELPER(kmmawt2)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t = 0;
    int64_t addop = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)(rs2_p[i] >> 16);
        v3 = rd_p[i];

        if(v1 == INT32_MIN && v2 == INT16_MIN) {
            addop = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            t = v1 * v2;
            addop = (int32_t)(t >> 15);
        }

        rd_p[i] = (int32_t)signed_saturate(env, v3 + addop, 32);
    }

    return rd;
}

target_ulong HELPER(kmmawt2_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t t = 0;
    int64_t addop = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = (int16_t)(rs2_p[i] >> 16);
        v3 = rd_p[i];

        if(v1 == INT32_MIN && v2 == INT16_MIN) {
            addop = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            t = v1 * v2;
            addop = (int32_t)((((uint64_t)t >> 14) + 1) >> 1);
        }
        
        rd_p[i] = (int32_t)signed_saturate(env, v3 + addop, 32);
    }

    return rd;
}

target_ulong HELPER(smbb16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        rd_p[i] = (int32_t)(v1 * v2);
    }

    return rd;
}

target_ulong HELPER(smbt16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)(rs2_p[i] >> 16);
        rd_p[i] = (int32_t)(v1 * v2);
    }

    return rd;
}

target_ulong HELPER(smtt16)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = (int16_t)(rs1_p[i] >> 16);
        v2 = (int16_t)(rs2_p[i] >> 16);
        rd_p[i] = (int32_t)(v1 * v2);
    }

    return rd;
}

target_ulong HELPER(kmda)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;
    int32_t v3 = 0;
    int32_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        if(rs1_p[i] == 0x80008000 && rs2_p[i] == 0x80008000) {
            rd_p[i] = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            v1 = (int16_t)(rs1_p[i] >> 16);
            v2 = (int16_t)(rs2_p[i] >> 16);
            v3 = (int16_t)rs1_p[i];
            v4 = (int16_t)rs2_p[i];
            rd_p[i] = (int32_t)(v1 * v2 + v3 * v4);
        }
    }

    return rd;
}

target_ulong HELPER(kmxda)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;
    int32_t v3 = 0;
    int32_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        if(rs1_p[i] == 0x80008000 && rs2_p[i] == 0x80008000) {
            rd_p[i] = INT32_MAX;
            env->vxsat = 0x1;
        } else {
            v1 = (int16_t)(rs1_p[i] >> 16);
            v2 = (int16_t)rs2_p[i];
            v3 = (int16_t)rs1_p[i];
            v4 = (int16_t)(rs2_p[i] >> 16);
            rd_p[i] = (int32_t)(v1 * v2 + v3 * v4);
        }
    }

    return rd;
}

target_ulong HELPER(smds)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;
    int32_t v3 = 0;
    int32_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = (int16_t)(rs1_p[i] >> 16);
        v2 = (int16_t)(rs2_p[i] >> 16);
        v3 = (int16_t)rs1_p[i];
        v4 = (int16_t)rs2_p[i];
        rd_p[i] = (int32_t)(v1 * v2 - v3 * v4);
    }

    return rd;
}

target_ulong HELPER(smdrs)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;
    int32_t v3 = 0;
    int32_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        v3 = (int16_t)(rs1_p[i] >> 16);
        v4 = (int16_t)(rs2_p[i] >> 16);
        rd_p[i] = (int32_t)(v1 * v2 - v3 * v4);
    }
    
    return rd;
}

target_ulong HELPER(smxds)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t v1 = 0;
    int32_t v2 = 0;
    int32_t v3 = 0;
    int32_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = (int16_t)(rs1_p[i] >> 16);
        v2 = (int16_t)rs2_p[i];
        v3 = (int16_t)rs1_p[i];
        v4 = (int16_t)(rs2_p[i] >> 16);
        rd_p[i] = (int32_t)(v1 * v2 - v3 * v4);
    }
    
    return rd;
}

target_ulong HELPER(kmabb)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t t = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t = rd_p[i];
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];

        rd_p[i] = (int32_t)signed_saturate(env, t + v1 * v2, 32);
    }

    return rd;
}

target_ulong HELPER(kmabt)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t t = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t = rd_p[i];
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)(rs2_p[i] >> 16);

        rd_p[i] = (int32_t)signed_saturate(env, t + v1 * v2, 32);
    }

    return rd;
}

target_ulong HELPER(kmatt)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t t = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t = rd_p[i];
        v1 = (int16_t)(rs1_p[i] >> 16);
        v2 = (int16_t)(rs2_p[i] >> 16);

        rd_p[i] = (int32_t)signed_saturate(env, t + v1 * v2, 32);
    }

    return rd;
}

target_ulong HELPER(kmada)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t t = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t = rd_p[i];
        v1 = (int16_t)(rs1_p[i] >> 16);
        v2 = (int16_t)(rs2_p[i] >> 16);
        v3 = (int16_t)rs1_p[i];
        v4 = (int16_t)rs2_p[i];

        rd_p[i] = (int32_t)signed_saturate(env, t + v1 * v2 + v3 * v4, 32);
    }

    return rd;
}

target_ulong HELPER(kmaxda)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t t = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t = rd_p[i];
        v1 = (int16_t)(rs1_p[i] >> 16);
        v2 = (int16_t)rs2_p[i];
        v3 = (int16_t)rs1_p[i];
        v4 = (int16_t)(rs2_p[i] >> 16);

        rd_p[i] = (int32_t)signed_saturate(env, t + v1 * v2 + v3 * v4, 32);
    }

    return rd;
}

target_ulong HELPER(kmads)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t t = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t = rd_p[i];
        v1 = (int16_t)(rs1_p[i] >> 16);
        v2 = (int16_t)(rs2_p[i] >> 16);
        v3 = (int16_t)rs1_p[i];
        v4 = (int16_t)rs2_p[i];

        rd_p[i] = (int32_t)signed_saturate(env, t + v1 * v2 - v3 * v4, 32);
    }

    return rd;
}

target_ulong HELPER(kmadrs)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t t = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t = rd_p[i];
        v1 = (int16_t)rs1_p[i];
        v2 = (int16_t)rs2_p[i];
        v3 = (int16_t)(rs1_p[i] >> 16);
        v4 = (int16_t)(rs2_p[i] >> 16);

        rd_p[i] = (int32_t)signed_saturate(env, t + v1 * v2 - v3 * v4, 32);
    }

    return rd;
}

target_ulong HELPER(kmaxds)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t t = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t = rd_p[i];
        v1 = (int16_t)(rs1_p[i] >> 16);
        v2 = (int16_t)rs2_p[i];
        v3 = (int16_t)rs1_p[i];
        v4 = (int16_t)(rs2_p[i] >> 16);

        rd_p[i] = (int32_t)signed_saturate(env, t + v1 * v2 - v3 * v4, 32);
    }

    return rd;
}

target_ulong HELPER(kmsda)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t t = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t = rd_p[i];
        v1 = (int16_t)(rs1_p[i] >> 16);
        v2 = (int16_t)(rs2_p[i] >> 16);
        v3 = (int16_t)rs1_p[i];
        v4 = (int16_t)rs2_p[i];

        rd_p[i] = (int32_t)signed_saturate(env, t - v1 * v2 - v3 * v4, 32);
    }

    return rd;
}

target_ulong HELPER(kmsxda)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t t = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;
    int64_t v4 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        t = rd_p[i];
        v1 = (int16_t)(rs1_p[i] >> 16);
        v2 = (int16_t)rs2_p[i];
        v3 = (int16_t)rs1_p[i];
        v4 = (int16_t)(rs2_p[i] >> 16);

        rd_p[i] = (int32_t)signed_saturate(env, t - v1 * v2 - v3 * v4, 32);
    }

    return rd;
}

target_ulong HELPER(smal_64)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs2_p = (int32_t*)&rs2;
    target_long v0 = (int16_t)(rs2_p[0]);
    target_long v1 = (int16_t)(rs2_p[0] >> 16);
    target_long v2 = (int16_t)(rs2_p[1]);
    target_long v3 = (int16_t)(rs2_p[1] >> 16);

    rd = (target_long)rs1 + v0 * v1 + v2 * v3;
    return rd;
}

target_ulong HELPER(sclip32)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1, shamt + 1);
    }

    return rd;
}

target_ulong HELPER(uclip32)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        target_long max = (1 << shamt) - 1;
        if(v1 > max) {
            v1 = max;
            env->vxsat = 0x1;
        } else if(v1 < 0) {
            v1 = 0;
            env->vxsat = 0x1;
        }
        rd_p[i] = v1;
    }

    return rd;
}

target_ulong HELPER(clrs32)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        int sign = v1 >> 31;
        int cnt = 0;
        
        for(int j = 30; j >= 0; j--) {
            if(((v1 >> j) & 1) == sign) {
                cnt++;
            } else {
                break;
            }
        }

        rd_p[i] = cnt;
    }

    return rd;
}

target_ulong HELPER(clz32)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    
    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        int cnt = 0;
        v1 = rs1_p[i];
        
        for(int j = 31; j >= 0; j--) {
            if((v1 >> j) == 0) {
                cnt++;
            } else {
                break;
            }
        }    

        rd_p[i] = cnt;
    }

    return rd;
}

target_ulong HELPER(pbsad)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    target_long t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t = (target_long)(v1 - v2);
        t = t > 0 ? t : -t;
        rd = rd + t;
    }

    return rd;
}

target_ulong HELPER(pbsada)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int8_t *rs1_p = (int8_t*)&rs1;
    int8_t *rs2_p = (int8_t*)&rs2;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    target_long t = 0;

    for(int i = 0; i < TARGET_LONG_SIZE; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        t = (target_long)(v1 - v2);
        t = t > 0 ? t : -t;
        rd = rd + t;
    }

    return rd;
}

target_ulong HELPER(smaqa)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rd_p = (int32_t*)&rd;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int64_t v0 = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v0 = (int8_t)(rs1_p[i]) * (int8_t)(rs2_p[i]);
        v1 = (int8_t)(rs1_p[i] >> 8) * (int8_t)(rs2_p[i] >> 8);
        v2 = (int8_t)(rs1_p[i] >> 16) * (int8_t)(rs2_p[i] >> 16);
        v3 = (int8_t)(rs1_p[i] >> 24) * (int8_t)(rs2_p[i] >> 24);
        rd_p[i] = rd_p[i] + v0 + v1 + v2 + v3;
    }

    return rd;
}

target_ulong HELPER(umaqa)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    uint32_t *rd_p = (uint32_t*)&rd;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint64_t v0 = 0;
    uint64_t v1 = 0;
    uint64_t v2 = 0;
    uint64_t v3 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v0 = (uint8_t)(rs1_p[i]) * (uint8_t)(rs2_p[i]);
        v1 = (uint8_t)(rs1_p[i] >> 8) * (uint8_t)(rs2_p[i] >> 8);
        v2 = (uint8_t)(rs1_p[i] >> 16) * (uint8_t)(rs2_p[i] >> 16);
        v3 = (uint8_t)(rs1_p[i] >> 24) * (uint8_t)(rs2_p[i] >> 24);
        rd_p[i] = rd_p[i] + v0 + v1 + v2 + v3;
    }

    return rd;
}

target_ulong HELPER(smaqa_su)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rd_p = (int32_t*)&rd;
    int32_t *rs1_p = (int32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    int64_t v0 = 0;
    int64_t v1 = 0;
    int64_t v2 = 0;
    int64_t v3 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v0 = (int8_t)(rs1_p[i]) * (uint8_t)(rs2_p[i]);
        v1 = (int8_t)(rs1_p[i] >> 8) * (uint8_t)(rs2_p[i] >> 8);
        v2 = (int8_t)(rs1_p[i] >> 16) * (uint8_t)(rs2_p[i] >> 16);
        v3 = (int8_t)(rs1_p[i] >> 24) * (uint8_t)(rs2_p[i] >> 24);
        rd_p[i] = rd_p[i] + v0 + v1 + v2 + v3;
    }

    return rd;
}

uint64_t HELPER(radd64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int64_t v1 = (int64_t)rs1;
    int64_t v2 = (int64_t)rs2;
    int64_t res = v1 + v2;
    int64_t over = (res ^ v1) & (res ^ v2) & INT64_MIN;

    /* With signed overflow, bit 64 is inverse of bit 63. */
    return (res >> 1) ^ over;
}

uint64_t HELPER(uradd64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    uint64_t res = rs1 + rs2;
    bool over = res < rs1;

    return over ? ((res >> 1) | INT64_MIN) : (res >> 1);
}

uint64_t HELPER(kadd64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int64_t v1 = (int64_t)rs1;
    int64_t v2 = (int64_t)rs2;
    int64_t res = v1 + v2;
    int64_t over = (res ^ v1) & (res ^ v2) & INT64_MIN;

    if(over) {
        res = v1 > 0 ? INT64_MAX : INT64_MIN;
        env->vxsat = 0x1;
    }

    return res;
}

uint64_t HELPER(ukadd64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    uint64_t res = rs1 + rs2;

    if (res < rs1) {
        res = UINT64_MAX;
        env->vxsat = 0x1;
    }

    return res;
}

uint64_t HELPER(sub64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int64_t v1 = (int64_t)rs1;
    int64_t v2 = (int64_t)rs2;

    return v1 - v2;
}

uint64_t HELPER(rsub64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int64_t v1 = (int64_t)rs1;
    int64_t v2 = (int64_t)rs2;
    int64_t res = v1 - v2;
    int64_t over = (res ^ v1) & (res ^ v2) & INT64_MIN;

    return (res >> 1) ^ over;
}

uint64_t HELPER(ursub64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    uint64_t res = rs1 - rs2;
    uint64_t over = (rs1 < rs2) ? INT64_MIN : 0;

    return (res >> 1) | over;
}

uint64_t HELPER(ksub64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int64_t v1 = (int64_t)rs1;
    int64_t v2 = (int64_t)rs2;
    int64_t res = v1 - v2;

    if ((res ^ v1) & (v1 ^ v2) & INT64_MIN) {
        res = v1 >= 0 ? INT64_MAX : INT64_MIN;
        env->vxsat = 0x1;
    }

    return res;
}

uint64_t HELPER(uksub64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    uint64_t res = rs1 - rs2;

    if (res > rs1) {
        res = 0;
        env->vxsat = 0x1;
    }

    return res;
}

uint64_t HELPER(smar64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    return rd + rs1_p[0] * rs2_p[0] + rs1_p[1] * rs2_p[1];
}

uint64_t HELPER(smsr64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    return rd - rs1_p[0] * rs2_p[0] - rs1_p[1] * rs2_p[1];
}

uint64_t HELPER(umar64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    return rd + rs1_p[0] * rs2_p[0] + rs1_p[1] * rs2_p[1];
}

uint64_t HELPER(umsr64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    return rd - rs1_p[0] * rs2_p[0] - rs1_p[1] * rs2_p[1];
}

uint64_t HELPER(kmar64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int64_t s_rd = (int64_t)rd;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    Int128 mul = (Int128)rs1_p[0] * (Int128)rs2_p[0] + (Int128)rs1_p[1] * (Int128)rs2_p[1];
    return signed_saturate64(env, (Int128)s_rd + mul);
}

uint64_t HELPER(kmsr64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int64_t s_rd = (int64_t)rd;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    Int128 mul = (Int128)rs1_p[0] * (Int128)rs2_p[0] + (Int128)rs1_p[1] * (Int128)rs2_p[1];
    return signed_saturate64(env, (Int128)s_rd - mul);
}

uint64_t HELPER(ukmar64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    Int128 mul = (Int128)rs1_p[0] * (Int128)rs2_p[0] + (Int128)rs1_p[1] * (Int128)rs2_p[1];
    return unsigned_saturate64(env, (Int128)rd + mul);
}

uint64_t HELPER(ukmsr64)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    __uint128_t mul = (__uint128_t)rs1_p[0] * (__uint128_t)rs2_p[0] + (__uint128_t)rs1_p[1] * (__uint128_t)rs2_p[1];
    return unsigned_saturate64(env, (__uint128_t)rd - mul);
}

uint64_t HELPER(smalbb)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int64_t mul = (int64_t)rs1_p[0] * (int64_t)rs2_p[0] + (int64_t)rs1_p[2] * (int64_t)rs2_p[2];
    return (int64_t)rd + mul;
}

uint64_t HELPER(smalbt)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int64_t mul = (int64_t)rs1_p[0] * (int64_t)rs2_p[1] + (int64_t)rs1_p[2] * (int64_t)rs2_p[3];
    return (int64_t)rd + mul;
}

uint64_t HELPER(smaltt)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int64_t mul = (int64_t)rs1_p[1] * (int64_t)rs2_p[1] + (int64_t)rs1_p[3] * (int64_t)rs2_p[3];
    return (int64_t)rd + mul;
}

uint64_t HELPER(smalda)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int64_t mul = (int64_t)rs1_p[0] * (int64_t)rs2_p[0] + (int64_t)rs1_p[1] * (int64_t)rs2_p[1] \
                + (int64_t)rs1_p[2] * (int64_t)rs2_p[2] + (int64_t)rs1_p[3] * (int64_t)rs2_p[3];
    return (int64_t)rd + mul;
}

uint64_t HELPER(smalxda)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int64_t mul = (int64_t)rs1_p[0] * (int64_t)rs2_p[1] + (int64_t)rs1_p[1] * (int64_t)rs2_p[0] \
                + (int64_t)rs1_p[2] * (int64_t)rs2_p[3] + (int64_t)rs1_p[3] * (int64_t)rs2_p[2];
    return (int64_t)rd + mul;
}

uint64_t HELPER(smalds)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int64_t mul = (int64_t)rs1_p[1] * (int64_t)rs2_p[1] - (int64_t)rs1_p[0] * (int64_t)rs2_p[0] \
                + (int64_t)rs1_p[3] * (int64_t)rs2_p[3] - (int64_t)rs1_p[2] * (int64_t)rs2_p[2];
    return (int64_t)rd + mul;
}

uint64_t HELPER(smaldrs)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int64_t mul = (int64_t)rs1_p[0] * (int64_t)rs2_p[0] - (int64_t)rs1_p[1] * (int64_t)rs2_p[1] \
                + (int64_t)rs1_p[2] * (int64_t)rs2_p[2] - (int64_t)rs1_p[3] * (int64_t)rs2_p[3];
    return (int64_t)rd + mul;
}

uint64_t HELPER(smalxds)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int64_t mul = (int64_t)rs1_p[1] * (int64_t)rs2_p[0] - (int64_t)rs1_p[0] * (int64_t)rs2_p[1] \
                + (int64_t)rs1_p[3] * (int64_t)rs2_p[2] - (int64_t)rs1_p[2] * (int64_t)rs2_p[3];
    return (int64_t)rd + mul;
}

uint64_t HELPER(smslda)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int64_t mul = (int64_t)rs1_p[0] * (int64_t)rs2_p[0] + (int64_t)rs1_p[1] * (int64_t)rs2_p[1] \
                + (int64_t)rs1_p[2] * (int64_t)rs2_p[2] + (int64_t)rs1_p[3] * (int64_t)rs2_p[3];
    return (int64_t)rd - mul;
}

uint64_t HELPER(smslxda)(CPURISCVState *env, uint64_t rs1, uint64_t rs2, uint64_t rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int64_t mul = (int64_t)rs1_p[0] * (int64_t)rs2_p[1] + (int64_t)rs1_p[1] * (int64_t)rs2_p[0] \
                + (int64_t)rs1_p[2] * (int64_t)rs2_p[3] + (int64_t)rs1_p[3] * (int64_t)rs2_p[2];
    return (int64_t)rd - mul;
}

target_ulong HELPER(kaddh)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    target_long res = (target_long)(rs1_p[0]) + (target_long)(rs2_p[0]);
    return signed_saturate(env, res, 16);
}

target_ulong HELPER(ksubh)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    target_long res = (target_long)(rs1_p[0]) - (target_long)(rs2_p[0]);
    return signed_saturate(env, res, 16);
}

target_ulong HELPER(khmbb)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t aop = rs1_p[0];
    int32_t bop = rs2_p[0];
    int32_t res = 0;

    if(aop == INT16_MIN && bop == INT16_MIN) {
        res = INT16_MAX;
        env->vxsat = 0x1;
    } else {
        res = (aop * bop) >> 15;
    }

    return (target_long)res;
}

target_ulong HELPER(khmbt)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t aop = rs1_p[0];
    int32_t bop = rs2_p[1];
    int32_t res = 0;

    if(aop == INT16_MIN && bop == INT16_MIN) {
        res = INT16_MAX;
        env->vxsat = 0x1;
    } else {
        res = (aop * bop) >> 15;
    }

    return (target_long)res;
}

target_ulong HELPER(khmtt)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t aop = rs1_p[1];
    int32_t bop = rs2_p[1];
    int32_t res = 0;

    if(aop == INT16_MIN && bop == INT16_MIN) {
        res = INT16_MAX;
        env->vxsat = 0x1;
    } else {
        res = (aop * bop) >> 15;
    }

    return (target_long)res;
}

target_ulong HELPER(ukaddh)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    target_ulong res = (target_ulong)(rs1_p[0]) + (target_ulong)(rs2_p[0]);
    int16_t t = unsigned_saturate(env, res, 16);
    return (target_long)t;
}

target_ulong HELPER(uksubh)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    uint16_t *rs1_p = (uint16_t*)&rs1;
    uint16_t *rs2_p = (uint16_t*)&rs2;
    target_ulong res = (target_ulong)(rs1_p[0]) - (target_ulong)(rs2_p[0]);
    int16_t t = unsigned_saturate(env, res, 16);
    return (target_long)t;
}

target_ulong HELPER(kaddw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int64_t res = (int64_t)(rs1_p[0]) + (int64_t)(rs2_p[0]);
    return signed_saturate(env, res, 32);
}

target_ulong HELPER(ukaddw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint64_t res = (uint64_t)(rs1_p[0]) + (uint64_t)(rs2_p[0]);
    int32_t t = unsigned_saturate(env, res, 32);
    return (target_long)t;
}

target_ulong HELPER(ksubw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int64_t res = (int64_t)(rs1_p[0]) - (int64_t)(rs2_p[0]);
    return signed_saturate(env, res, 32);
}

target_ulong HELPER(uksubw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint64_t res = (uint64_t)(rs1_p[0]) - (uint64_t)(rs2_p[0]);
    int32_t t = unsigned_saturate(env, res, 32);
    return (target_long)t;
}

target_ulong HELPER(kdmbb)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t aop = rs1_p[0];
    int32_t bop = rs2_p[0];
    int32_t res = 0;

    if(aop == INT16_MIN && bop == INT16_MIN) {
        res = INT32_MAX;
        env->vxsat = 0x1;
    } else {
        res = (aop * bop) << 1;
    }

    return (target_long)res;
}

target_ulong HELPER(kdmbt)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t aop = rs1_p[0];
    int32_t bop = rs2_p[1];
    int32_t res = 0;

    if(aop == INT16_MIN && bop == INT16_MIN) {
        res = INT32_MAX;
        env->vxsat = 0x1;
    } else {
        res = (aop * bop) << 1;
    }

    return (target_long)res;
}

target_ulong HELPER(kdmtt)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t aop = rs1_p[1];
    int32_t bop = rs2_p[1];
    int32_t res = 0;

    if(aop == INT16_MIN && bop == INT16_MIN) {
        res = INT32_MAX;
        env->vxsat = 0x1;
    } else {
        res = (aop * bop) << 1;
    }

    return (target_long)res;
}

target_ulong HELPER(kslraw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_long rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    target_ulong shamt = rs2 & 0x3F;
    int64_t v1 = rs1_p[0];

    if(shamt & 0x20) {
        shamt = 32 - (shamt & 0x1F);
        if(shamt == 32)
            shamt = 31;
        rd = v1 >> shamt;
    } else {
        rd = signed_saturate(env, v1 << shamt, 32);
    }

    return rd;
}

target_ulong HELPER(kslraw_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_long rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    target_ulong shamt = rs2 & 0x3F;
    int64_t v1 = rs1_p[0];

    if(shamt & 0x20) {
        shamt = 32 - (shamt & 0x1F);
        if(shamt == 32)
            shamt = 31;
        rd = ((v1 >> (shamt - 1)) + 1) >> 1;
    } else {
        rd = signed_saturate(env, v1 << shamt, 32);
    }

    return rd;
}

target_ulong HELPER(ksllw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    target_ulong shamt = rs2 & 0x1F;
    int64_t v1 = rs1_p[0];

    rd = signed_saturate(env, v1 << shamt, 32);

    return rd;
}

target_ulong HELPER(kslliw)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int64_t v1 = rs1_p[0];

    rd = signed_saturate(env, v1 << shamt, 32);

    return rd;
}

target_ulong HELPER(kdmabb)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t aop = rs1_p[0];
    int64_t bop = rs2_p[0];
    int64_t res = 0;
    int64_t c = rd_p[0];

    if(aop == INT16_MIN && bop == INT16_MIN) {
        res = INT32_MAX;
        env->vxsat = 0x1;
    } else {
        res = (aop * bop) << 1;
    }

    return signed_saturate(env, res + c, 32);
}

target_ulong HELPER(kdmabt)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t aop = rs1_p[0];
    int64_t bop = rs2_p[1];
    int64_t res = 0;
    int64_t c = rd_p[0];

    if(aop == INT16_MIN && bop == INT16_MIN) {
        res = INT32_MAX;
        env->vxsat = 0x1;
    } else {
        res = (aop * bop) << 1;
    }

    return signed_saturate(env, res + c, 32);
}

target_ulong HELPER(kdmatt)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int16_t *rs1_p = (int16_t*)&rs1;
    int16_t *rs2_p = (int16_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int64_t aop = rs1_p[1];
    int64_t bop = rs2_p[1];
    int64_t res = 0;
    int64_t c = rd_p[0];

    if(aop == INT16_MIN && bop == INT16_MIN) {
        res = INT32_MAX;
        env->vxsat = 0x1;
    } else {
        res = (aop * bop) << 1;
    }

    return signed_saturate(env, res + c, 32);
}

target_ulong HELPER(kabsw)(CPURISCVState *env, target_ulong rs1)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int64_t v1 = rs1_p[0];

    if(v1 == INT32_MIN) {
        v1 = INT32_MAX;
        env->vxsat = 0x1;
    } else if(v1 < 0) {
        v1 = -v1;
    }
    rd = v1;

    return rd;
}

target_ulong HELPER(raddw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int64_t v1 = rs1_p[0];
    int64_t v2 = rs2_p[0];

    return (v1 + v2) >> 1;
}

target_ulong HELPER(uraddw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint64_t v1 = rs1_p[0];
    uint64_t v2 = rs2_p[0];

    return (v1 + v2) >> 1;
}

target_ulong HELPER(rsubw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int64_t v1 = rs1_p[0];
    int64_t v2 = rs2_p[0];

    return (v1 - v2) >> 1;
}

target_ulong HELPER(ursubw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint64_t v1 = rs1_p[0];
    uint64_t v2 = rs2_p[0];

    return (v1 - v2) >> 1;
}

target_ulong HELPER(maxw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int64_t v1 = rs1_p[0];
    int64_t v2 = rs2_p[0];

    return v1 >= v2 ? v1 : v2;
}

target_ulong HELPER(minw)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int64_t v1 = rs1_p[0];
    int64_t v2 = rs2_p[0];

    return v1 >= v2 ? v2 : v1;
}

target_ulong HELPER(ave)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_long v1 = rs1;
    target_long v2 = rs2;

    Int128 res = (Int128)v1 + (Int128)v2 + 1;

    return res >> 1;
}

target_ulong HELPER(sra_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    target_ulong shamt = rs2 & 0x1F;
    target_long v1 = rs1;

    if(shamt == 0) {
        rd = rs1;
    } else {
        rd = ((v1 >> (shamt - 1)) + 1) >> 1;
    }

    return rd;
}

target_ulong HELPER(srai_u)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    target_long v1 = rs1;

    if(shamt == 0) {
        rd = rs1;
    } else {
        rd = ((v1 >> (shamt - 1)) + 1) >> 1;
    }

    return rd;
}

target_ulong HELPER(bitrev)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    uint8_t shift = riscv_cpu_xlen(env) == 32 ? (rs2 & 0x1f) : (rs2 & 0x3f);
    return  revbit64(rs1) >> (64 - shift - 1);
}

target_ulong HELPER(bitrevi)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    return  revbit64(rs1) >> (64 - shamt - 1);
}

target_ulong HELPER(wext)(CPURISCVState *env, uint64_t rs1, target_ulong rs2)
{
    return  (target_long)((int32_t)(rs1 >> rs2));
}

target_ulong HELPER(cmix)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rs3)
{
    return (rs2 & rs1) | (~rs2 & rs3);
}

target_ulong HELPER(insb)(CPURISCVState *env, target_ulong rs1, target_ulong shamt, target_ulong rd)
{
    uint8_t *rs1_p = (uint8_t*)&rs1;
    uint8_t *rd_p = (uint8_t*)&rd;
    rd_p[shamt] = rs1_p[shamt];
    return rd;
}

target_ulong HELPER(maddr32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t res = rd_p[0] + rs1_p[0] * rs2_p[0];
    return (target_long)res;
}

target_ulong HELPER(msubr32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2, target_ulong rd)
{
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    int32_t res = rd_p[0] - rs1_p[0] * rs2_p[0];
    return (target_long)res;
}

target_ulong HELPER(add32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)(v1 + v2);
    }

    return rd;
}

target_ulong HELPER(radd32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(uradd32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)((v1 + v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(kadd32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 + v2, 32);
    }

    return rd;
}

target_ulong HELPER(ukadd32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)unsigned_saturate(env, v1 + v2, 32);
    }

    return rd;
}

target_ulong HELPER(sub32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)(v1 - v2);
    }

    return rd;
}

target_ulong HELPER(rsub32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(ursub32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)((v1 - v2) >> 1);
    }

    return rd;
}

target_ulong HELPER(ksub32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 - v2, 32);
    }

    return rd;
}

target_ulong HELPER(uksub32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)unsigned_saturate(env, v1 - v2, 32);
    }

    return rd;
}

target_ulong HELPER(cras32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)(v1 + v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)(v1 - v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(rcras32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)((v1 + v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(urcras32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (uint32_t)((v1 + v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint32_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kcras32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)signed_saturate(env, v1 + v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)signed_saturate(env, v1 - v2, 32);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukcras32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (uint32_t)unsigned_saturate(env, v1 + v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint32_t)unsigned_saturate(env, v1 - v2, 32);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(crsa32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)(v1 - v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)(v1 + v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(rcrsa32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)((v1 - v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)((v1 + v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(urcrsa32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (uint32_t)((v1 - v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint32_t)((v1 + v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kcrsa32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (int32_t)signed_saturate(env, v1 - v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (int32_t)signed_saturate(env, v1 + v2, 32);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukcrsa32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i - 1];
        rd_p[i] = (uint32_t)unsigned_saturate(env, v1 - v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i];
        rd_p[i - 1] = (uint32_t)unsigned_saturate(env, v1 + v2, 32);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(stas32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)(v1 + v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int32_t)(v1 - v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(rstas32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)((v1 + v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int32_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(urstas32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)((v1 + v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint32_t)((v1 - v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kstas32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 + v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int32_t)signed_saturate(env, v1 - v2, 32);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukstas32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)unsigned_saturate(env, v1 + v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint32_t)unsigned_saturate(env, v1 - v2, 32);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(stsa32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)(v1 - v2);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int32_t)(v1 + v2);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(urstsa32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)((v1 - v2) >> 1);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint32_t)((v1 + v2) >> 1);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(kstsa32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rs2_p = (int32_t*)&rs2;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;
    target_long v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 - v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (int32_t)signed_saturate(env, v1 + v2, 32);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(ukstsa32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rs2_p = (uint32_t*)&rs2;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;
    target_ulong v2 = 0;
    int i = 1;

    while(i < TARGET_LONG_SIZE / 4) {
        v1 = rs1_p[i];
        v2 = rs2_p[i];
        rd_p[i] = (uint32_t)unsigned_saturate(env, v1 - v2, 32);
        v1 = rs1_p[i - 1];
        v2 = rs2_p[i - 1];
        rd_p[i - 1] = (uint32_t)unsigned_saturate(env, v1 + v2, 32);
        i = i + 2;
    }

    return rd;
}

target_ulong HELPER(sra32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_ulong shamt = rs2 & 0x1F;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i]; 
        rd_p[i] = (int32_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(srai32)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(sra32_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_ulong shamt = rs2 & 0x1F;
    target_long v1 = 0;

    if(shamt == 0) {
        rd = rs1;
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int32_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srai32_u)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;

    if(shamt == 0) {
        rd = rs1;
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int32_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(srl32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong shamt = rs2 & 0x1F;
    target_ulong v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint32_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(srli32)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (uint32_t)(v1 >> shamt);
    }

    return rd;
}

target_ulong HELPER(srl32_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    uint32_t *rs1_p = (uint32_t*)&rs1;
    uint32_t *rd_p = (uint32_t*)&rd;
    target_ulong shamt = rs2 & 0x1F;
    target_ulong v1 = 0;

    if(shamt == 0) {
        rd = rs1;
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (uint32_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    }

    return rd;
}

target_ulong HELPER(sll32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_ulong shamt = rs2 & 0x1F;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 << shamt);
    }

    return rd;
}

target_ulong HELPER(slli32)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)(v1 << shamt);
    }

    return rd;
}

target_ulong HELPER(ksll32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_ulong shamt = rs2 & 0x1F;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 << shamt, 32);
    }

    return rd;
}

target_ulong HELPER(kslli32)(CPURISCVState *env, target_ulong rs1, target_ulong shamt)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_long v1 = 0;

    for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
        v1 = rs1_p[i];
        rd_p[i] = (int32_t)signed_saturate(env, v1 << shamt, 32);
    }

    return rd;
}

target_ulong HELPER(kslra32)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_ulong shamt = rs2 & 0x3F;
    target_long v1 = 0;

    if(shamt & 0x20) {
        shamt = 32 - (shamt & 0x1F);
        if(shamt == 32)
            shamt = 31;

        for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int32_t)(v1 >> shamt);
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int32_t)signed_saturate(env, v1 << shamt, 32);
        }
    }

    return rd;
}

target_ulong HELPER(kslra32_u)(CPURISCVState *env, target_ulong rs1, target_ulong rs2)
{
    target_ulong rd = 0;
    int32_t *rs1_p = (int32_t*)&rs1;
    int32_t *rd_p = (int32_t*)&rd;
    target_ulong shamt = rs2 & 0x3F;
    target_long v1 = 0;

    if(shamt & 0x20) {
        shamt = 32 - (shamt & 0x1F);
        if(shamt == 32)
            shamt = 31;

        for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int32_t)(((v1 >> (shamt - 1)) + 1) >> 1);
        }
    } else {
        for(int i = 0; i < TARGET_LONG_SIZE / 4; i++) {
            v1 = rs1_p[i];
            rd_p[i] = (int32_t)signed_saturate(env, v1 << shamt, 32);
        }
    }

    return rd;
}
