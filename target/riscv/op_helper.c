/*
 * RISC-V Emulation Helpers for QEMU.
 *
 * Copyright (c) 2016-2017 Sagar Karandikar, sagark@eecs.berkeley.edu
 * Copyright (c) 2017-2018 SiFive, Inc.
 * Copyright (c) 2022      VRULL GmbH
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
#include "internals.h"
#include "qemu/main-loop.h"
#include "exec/exec-all.h"
#include "exec/helper-proto.h"
#include "exec/cpu_ldst.h"
#include "qemu/log.h"

#if !defined(CONFIG_USER_ONLY)
#include "hw/intc/riscv_clic.h"
#endif

void helper_tb_trace(CPURISCVState *env, target_ulong tb_pc)
{
    int trace_index = env->trace_index % TB_TRACE_NUM;
    env->trace_info[trace_index].tb_pc = tb_pc;
    env->trace_info[trace_index].notjmp = false;
    env->trace_index++;
    if (env->jcount_enable == 0) {
        qemu_log_mask(CPU_TB_TRACE, "0x" TARGET_FMT_lx "\n", tb_pc);
    } else if ((tb_pc > env->jcount_start) &&
                (tb_pc < env->jcount_end)) {
        qemu_log_mask(CPU_TB_TRACE, "0x" TARGET_FMT_lx "\n", tb_pc);
    }
}

void helper_tag_pctrace(CPURISCVState *env, target_ulong tb_pc)
{
    if (env->trace_index < 1) {
        return;
    }
    int trace_index = (env->trace_index -1) % TB_TRACE_NUM;
    if (env->trace_info[trace_index].tb_pc == tb_pc) {
        env->trace_info[trace_index].notjmp = true;
    }
}

#ifdef CONFIG_USER_ONLY
extern long long total_jcount;
void helper_jcount(CPURISCVState *env, target_ulong tb_pc, uint32_t icount)
{
    if ((tb_pc >= env->jcount_start) && (tb_pc < env->jcount_end)) {
        total_jcount += icount;
    }
}
#else
void helper_jcount(CPURISCVState *env, target_ulong tb_pc, uint32_t icount)
{
}
#endif

/* Exceptions processing helpers */
G_NORETURN void riscv_raise_exception(CPURISCVState *env,
                                      uint32_t exception, uintptr_t pc)
{
    CPUState *cs = env_cpu(env);
    cs->exception_index = exception;
    cpu_loop_exit_restore(cs, pc);
}

void helper_raise_exception(CPURISCVState *env, uint32_t exception)
{
    riscv_raise_exception(env, exception, 0);
}

target_ulong helper_csrr(CPURISCVState *env, int csr)
{
    /*
     * The seed CSR must be accessed with a read-write instruction. A
     * read-only instruction such as CSRRS/CSRRC with rs1=x0 or CSRRSI/
     * CSRRCI with uimm=0 will raise an illegal instruction exception.
     */
    if (csr == CSR_SEED) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    target_ulong val = 0;
    RISCVException ret = riscv_csrrw(env, csr, &val, 0, 0);

    if (ret != RISCV_EXCP_NONE) {
        riscv_raise_exception(env, ret, GETPC());
    }
    return val;
}

void helper_csrw(CPURISCVState *env, int csr, target_ulong src)
{
    target_ulong mask = env->xl == MXL_RV32 ? UINT32_MAX : (target_ulong)-1;
    RISCVException ret = riscv_csrrw(env, csr, NULL, src, mask);

    if (ret != RISCV_EXCP_NONE) {
        riscv_raise_exception(env, ret, GETPC());
    }
}

target_ulong helper_csrrw(CPURISCVState *env, int csr,
                          target_ulong src, target_ulong write_mask)
{
    target_ulong val = 0;
    RISCVException ret = riscv_csrrw(env, csr, &val, src, write_mask);

    if (ret != RISCV_EXCP_NONE) {
        riscv_raise_exception(env, ret, GETPC());
    }
    return val;
}

target_ulong helper_csrr_i128(CPURISCVState *env, int csr)
{
    Int128 rv = int128_zero();
    RISCVException ret = riscv_csrrw_i128(env, csr, &rv,
                                          int128_zero(),
                                          int128_zero());

    if (ret != RISCV_EXCP_NONE) {
        riscv_raise_exception(env, ret, GETPC());
    }

    env->retxh = int128_gethi(rv);
    return int128_getlo(rv);
}

void helper_csrw_i128(CPURISCVState *env, int csr,
                      target_ulong srcl, target_ulong srch)
{
    RISCVException ret = riscv_csrrw_i128(env, csr, NULL,
                                          int128_make128(srcl, srch),
                                          UINT128_MAX);

    if (ret != RISCV_EXCP_NONE) {
        riscv_raise_exception(env, ret, GETPC());
    }
}

target_ulong helper_csrrw_i128(CPURISCVState *env, int csr,
                               target_ulong srcl, target_ulong srch,
                               target_ulong maskl, target_ulong maskh)
{
    Int128 rv = int128_zero();
    RISCVException ret = riscv_csrrw_i128(env, csr, &rv,
                                          int128_make128(srcl, srch),
                                          int128_make128(maskl, maskh));

    if (ret != RISCV_EXCP_NONE) {
        riscv_raise_exception(env, ret, GETPC());
    }

    env->retxh = int128_gethi(rv);
    return int128_getlo(rv);
}

/*
 * check_zicbo_envcfg
 *
 * Raise virtual exceptions and illegal instruction exceptions for
 * Zicbo[mz] instructions based on the settings of [mhs]envcfg as
 * specified in section 2.5.1 of the CMO specification.
 */
static void check_zicbo_envcfg(CPURISCVState *env, target_ulong envbits,
                                uintptr_t ra)
{
#ifndef CONFIG_USER_ONLY
    if ((env->priv < PRV_M) && !get_field(env->menvcfg, envbits)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }

    if (env->virt_enabled &&
        (((env->priv <= PRV_S) && !get_field(env->henvcfg, envbits)) ||
         ((env->priv < PRV_S) && !get_field(env->senvcfg, envbits)))) {
        riscv_raise_exception(env, RISCV_EXCP_VIRT_INSTRUCTION_FAULT, ra);
    }

    if ((env->priv < PRV_S) && !get_field(env->senvcfg, envbits)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }
#endif
}

void helper_cbo_zero(CPURISCVState *env, target_ulong address)
{
    RISCVCPU *cpu = env_archcpu(env);
    uint16_t cbozlen = cpu->cfg.cboz_blocksize;
    int mmu_idx = cpu_mmu_index(env, false);
    uintptr_t ra = GETPC();
    void *mem;

    check_zicbo_envcfg(env, MENVCFG_CBZE, ra);

    /* Mask off low-bits to align-down to the cache-block. */
    address &= ~(cbozlen - 1);

    /*
     * cbo.zero requires MMU_DATA_STORE access. Do a probe_write()
     * to raise any exceptions, including PMP.
     */
    mem = probe_write(env, address, cbozlen, mmu_idx, ra);

    if (likely(mem)) {
        memset(mem, 0, cbozlen);
    } else {
        /*
         * This means that we're dealing with an I/O page. Section 4.2
         * of cmobase v1.0.1 says:
         *
         * "Cache-block zero instructions store zeros independently
         * of whether data from the underlying memory locations are
         * cacheable."
         *
         * Write zeros in address + cbozlen regardless of not being
         * a RAM page.
         */
        for (int i = 0; i < cbozlen; i++) {
            cpu_stb_mmuidx_ra(env, address + i, 0, mmu_idx, ra);
        }
    }
}

/*
 * check_zicbom_access
 *
 * Check access permissions (LOAD, STORE or FETCH as specified in
 * section 2.5.2 of the CMO specification) for Zicbom, raising
 * either store page-fault (non-virtualized) or store guest-page
 * fault (virtualized).
 */
static void check_zicbom_access(CPURISCVState *env,
                                target_ulong address,
                                uintptr_t ra)
{
    RISCVCPU *cpu = env_archcpu(env);
    int mmu_idx = cpu_mmu_index(env, false);
    uint16_t cbomlen = cpu->cfg.cbom_blocksize;
    void *phost;
    int ret;

    /* Mask off low-bits to align-down to the cache-block. */
    address &= ~(cbomlen - 1);

    /*
     * Section 2.5.2 of cmobase v1.0.1:
     *
     * "A cache-block management instruction is permitted to
     * access the specified cache block whenever a load instruction
     * or store instruction is permitted to access the corresponding
     * physical addresses. If neither a load instruction nor store
     * instruction is permitted to access the physical addresses,
     * but an instruction fetch is permitted to access the physical
     * addresses, whether a cache-block management instruction is
     * permitted to access the cache block is UNSPECIFIED."
     */
    ret = probe_access_flags(env, address, cbomlen, MMU_DATA_LOAD,
                             mmu_idx, true, &phost, ra);
    if (ret != TLB_INVALID_MASK) {
        /* Success: readable */
        return;
    }

    /*
     * Since not readable, must be writable. On failure, store
     * fault/store guest amo fault will be raised by
     * riscv_cpu_tlb_fill(). PMP exceptions will be caught
     * there as well.
     */
    probe_write(env, address, cbomlen, mmu_idx, ra);
}

void helper_cbo_clean_flush(CPURISCVState *env, target_ulong address)
{
    uintptr_t ra = GETPC();
    check_zicbo_envcfg(env, MENVCFG_CBCFE, ra);
    check_zicbom_access(env, address, ra);

    /* We don't emulate the cache-hierarchy, so we're done. */
}

void helper_cbo_inval(CPURISCVState *env, target_ulong address)
{
    uintptr_t ra = GETPC();
    check_zicbo_envcfg(env, MENVCFG_CBIE, ra);
    check_zicbom_access(env, address, ra);

    /* We don't emulate the cache-hierarchy, so we're done. */
}

#ifndef CONFIG_USER_ONLY

void helper_switch_context_xl(CPURISCVState *env)
{
    RISCVMXL xl = env->xl;
    if (xl < env->misa_mxl_max) {
        switch (xl) {
        case MXL_RV32:
            for (int i = 1; i < 32; i++) {
                env->gpr[i] = (int32_t)env->gpr[i];
            }
            env->pc = (int32_t)env->pc;
            /*
             * For the read-only bits of the previous-width CSR, the bits at the
             * same positions in the temporary register are set to zeros.
             */
            if ((env->priv == PRV_U) && (env->misa_ext & RVV)) {
                env->vl = 0;
                env->vtype = 0;
            }
            break;
        default:
            break;
        }
    }
}

target_ulong helper_sret(CPURISCVState *env)
{
    uint64_t mstatus;
    target_ulong prev_priv, prev_virt;

    if (!(env->priv >= PRV_S)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    target_ulong retpc = env->sepc;
    if (!riscv_has_ext(env, RVC) && (retpc & 0x3)) {
        riscv_raise_exception(env, RISCV_EXCP_INST_ADDR_MIS, GETPC());
    }

    if (get_field(env->mstatus, MSTATUS_TSR) && !(env->priv >= PRV_M)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    if (env->virt_enabled && get_field(env->hstatus, HSTATUS_VTSR)) {
        riscv_raise_exception(env, RISCV_EXCP_VIRT_INSTRUCTION_FAULT, GETPC());
    }

    mstatus = env->mstatus;
    prev_priv = get_field(mstatus, MSTATUS_SPP);
    mstatus = set_field(mstatus, MSTATUS_SIE,
                        get_field(mstatus, MSTATUS_SPIE));
    mstatus = set_field(mstatus, MSTATUS_SPIE, 1);
    mstatus = set_field(mstatus, MSTATUS_SPP, PRV_U);
    if (env->priv_ver >= PRIV_VERSION_1_12_0) {
        mstatus = set_field(mstatus, MSTATUS_MPRV, 0);
    }
    env->mstatus = mstatus;

    if (riscv_has_ext(env, RVH) && !env->virt_enabled) {
        /* We support Hypervisor extensions and virtulisation is disabled */
        target_ulong hstatus = env->hstatus;

        prev_virt = get_field(hstatus, HSTATUS_SPV);

        hstatus = set_field(hstatus, HSTATUS_SPV, 0);

        env->hstatus = hstatus;

        if (prev_virt) {
            riscv_cpu_swap_hypervisor_regs(env);
        }

        riscv_cpu_set_virt_enabled(env, prev_virt);
    } else {
        if (riscv_clic_is_clic_mode(env)) {
            CPUState *cs = env_cpu(env);
            target_ulong spil = get_field(env->scause, SCAUSE_SPIL);
            env->mintstatus = set_field(env->mintstatus, MINTSTATUS_SIL, spil);
            env->scause = set_field(env->scause, SCAUSE_SPIE, 0);
            env->scause = set_field(env->scause, SCAUSE_SPP, PRV_U);
            qemu_mutex_lock_iothread();
            riscv_clic_get_next_interrupt(env->clic, cs->cpu_index);
            qemu_mutex_unlock_iothread();
        }
    }

    riscv_cpu_set_mode(env, prev_priv);
    return retpc;
}

static target_ulong do_excp_return(CPURISCVState *env, target_ulong ra)
{
    if (!(env->priv >= PRV_M)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }

    target_ulong retpc = env->mepc;
    if (!riscv_has_ext(env, RVC) && (retpc & 0x3)) {
        riscv_raise_exception(env, RISCV_EXCP_INST_ADDR_MIS, ra);
    }

    /* if CLIC mode, copy mcause.mpil into minstatus.mil */
    if ((env->mtvec & 0b111110) == 0b000010) {
        target_ulong mpil = get_field(env->mcause, MCAUSE_MPIL);
        env->mintstatus = set_field(env->mintstatus, MINTSTATUS_MIL, mpil);
        if ((mpil == 0) && (env->mexstatus & MEXSTATUS_SPSWAP)
            && ((target_long)env->mcause < 0)) {
            target_ulong tmp = env->mscratch;
            env->mscratch = env->gpr[2];
            env->gpr[2] = tmp;
        }
    }

    uint64_t mstatus = env->mstatus;
    target_ulong prev_priv = get_field(mstatus, MSTATUS_MPP);

    if (riscv_cpu_cfg(env)->pmp &&
        !pmp_get_num_rules(env) && (prev_priv != PRV_M)) {
        riscv_raise_exception(env, RISCV_EXCP_INST_ACCESS_FAULT, ra);
    }

    target_ulong prev_virt = get_field(env->mstatus, MSTATUS_MPV);
    mstatus = set_field(mstatus, MSTATUS_MIE,
                        get_field(mstatus, MSTATUS_MPIE));
    mstatus = set_field(mstatus, MSTATUS_MPIE, 1);
    mstatus = set_field(mstatus, MSTATUS_MPP,
                        riscv_has_ext(env, RVU) ? PRV_U : PRV_M);
    mstatus = set_field(mstatus, MSTATUS_MPV, 0);
    if ((env->priv_ver >= PRIV_VERSION_1_12_0) && (prev_priv != PRV_M)) {
        mstatus = set_field(mstatus, MSTATUS_MPRV, 0);
    }
    env->mstatus = mstatus;
    riscv_cpu_set_mode(env, prev_priv);

    if (riscv_has_ext(env, RVH)) {
        if (prev_virt) {
            riscv_cpu_swap_hypervisor_regs(env);
        }

        riscv_cpu_set_virt_enabled(env, prev_virt);
    }

    if (riscv_clic_is_clic_mode(env)) {
        CPUState *cs = env_cpu(env);
        target_ulong mpil = get_field(env->mcause, MCAUSE_MPIL);
        env->mintstatus = set_field(env->mintstatus, MINTSTATUS_MIL, mpil);
        env->mcause = set_field(env->mcause, MCAUSE_MPIE, 0);
        env->mcause = set_field(env->mcause, MCAUSE_MPP, PRV_U);
        qemu_mutex_lock_iothread();
        riscv_clic_get_next_interrupt(env->clic, cs->cpu_index);
        qemu_mutex_unlock_iothread();
    }
    env->excp_vld = 0;
    return retpc;
}

target_ulong helper_mret(CPURISCVState *env)
{
    return do_excp_return(env, GETPC());
}

void helper_wfi(CPURISCVState *env)
{
    CPUState *cs = env_cpu(env);
    bool rvs = riscv_has_ext(env, RVS);
    bool prv_u = env->priv == PRV_U;
    bool prv_s = env->priv == PRV_S;

    if (((prv_s || (!rvs && prv_u)) && get_field(env->mstatus, MSTATUS_TW)) ||
        (rvs && prv_u && !env->virt_enabled)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    } else if (env->virt_enabled &&
               (prv_u || (prv_s && get_field(env->hstatus, HSTATUS_VTW)))) {
        riscv_raise_exception(env, RISCV_EXCP_VIRT_INSTRUCTION_FAULT, GETPC());
    } else {
        cs->halted = 1;
        cs->exception_index = EXCP_HLT;
        cpu_loop_exit(cs);
    }
}

void helper_tlb_flush(CPURISCVState *env)
{
    CPUState *cs = env_cpu(env);
    if (!env->virt_enabled &&
        (env->priv == PRV_U ||
         (env->priv == PRV_S && get_field(env->mstatus, MSTATUS_TVM)))) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    } else if (env->virt_enabled &&
               (env->priv == PRV_U || get_field(env->hstatus, HSTATUS_VTVM))) {
        riscv_raise_exception(env, RISCV_EXCP_VIRT_INSTRUCTION_FAULT, GETPC());
    } else {
        tlb_flush(cs);
    }
}

void helper_tlb_flush_all(CPURISCVState *env)
{
    CPUState *cs = env_cpu(env);
    tlb_flush_all_cpus_synced(cs);
}

void helper_hyp_tlb_flush(CPURISCVState *env)
{
    CPUState *cs = env_cpu(env);

    if (env->virt_enabled) {
        riscv_raise_exception(env, RISCV_EXCP_VIRT_INSTRUCTION_FAULT, GETPC());
    }

    if (env->priv == PRV_M ||
        (env->priv == PRV_S && !env->virt_enabled)) {
        tlb_flush(cs);
        return;
    }

    riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
}

void helper_hyp_gvma_tlb_flush(CPURISCVState *env)
{
    if (env->priv == PRV_S && !env->virt_enabled &&
        get_field(env->mstatus, MSTATUS_TVM)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC());
    }

    helper_hyp_tlb_flush(env);
}

static int check_access_hlsv(CPURISCVState *env, bool x, uintptr_t ra)
{
    if (env->priv == PRV_M) {
        /* always allowed */
    } else if (env->virt_enabled) {
        riscv_raise_exception(env, RISCV_EXCP_VIRT_INSTRUCTION_FAULT, ra);
    } else if (env->priv == PRV_U && !get_field(env->hstatus, HSTATUS_HU)) {
        riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, ra);
    }

    int mode = get_field(env->hstatus, HSTATUS_SPVP);
    if (!x && mode == PRV_S && get_field(env->vsstatus, MSTATUS_SUM)) {
        mode = MMUIdx_S_SUM;
    }
    return mode | MMU_2STAGE_BIT;
}

target_ulong helper_hyp_hlv_bu(CPURISCVState *env, target_ulong addr)
{
    uintptr_t ra = GETPC();
    int mmu_idx = check_access_hlsv(env, false, ra);
    MemOpIdx oi = make_memop_idx(MO_UB, mmu_idx);

    return cpu_ldb_mmu(env, addr, oi, ra);
}

target_ulong helper_hyp_hlv_hu(CPURISCVState *env, target_ulong addr)
{
    uintptr_t ra = GETPC();
    int mmu_idx = check_access_hlsv(env, false, ra);
    MemOpIdx oi = make_memop_idx(MO_TEUW, mmu_idx);

    return cpu_ldw_mmu(env, addr, oi, ra);
}

target_ulong helper_hyp_hlv_wu(CPURISCVState *env, target_ulong addr)
{
    uintptr_t ra = GETPC();
    int mmu_idx = check_access_hlsv(env, false, ra);
    MemOpIdx oi = make_memop_idx(MO_TEUL, mmu_idx);

    return cpu_ldl_mmu(env, addr, oi, ra);
}

target_ulong helper_hyp_hlv_d(CPURISCVState *env, target_ulong addr)
{
    uintptr_t ra = GETPC();
    int mmu_idx = check_access_hlsv(env, false, ra);
    MemOpIdx oi = make_memop_idx(MO_TEUQ, mmu_idx);

    return cpu_ldq_mmu(env, addr, oi, ra);
}

void helper_hyp_hsv_b(CPURISCVState *env, target_ulong addr, target_ulong val)
{
    uintptr_t ra = GETPC();
    int mmu_idx = check_access_hlsv(env, false, ra);
    MemOpIdx oi = make_memop_idx(MO_UB, mmu_idx);

    cpu_stb_mmu(env, addr, val, oi, ra);
}

void helper_hyp_hsv_h(CPURISCVState *env, target_ulong addr, target_ulong val)
{
    uintptr_t ra = GETPC();
    int mmu_idx = check_access_hlsv(env, false, ra);
    MemOpIdx oi = make_memop_idx(MO_TEUW, mmu_idx);

    cpu_stw_mmu(env, addr, val, oi, ra);
}

void helper_hyp_hsv_w(CPURISCVState *env, target_ulong addr, target_ulong val)
{
    uintptr_t ra = GETPC();
    int mmu_idx = check_access_hlsv(env, false, ra);
    MemOpIdx oi = make_memop_idx(MO_TEUL, mmu_idx);

    cpu_stl_mmu(env, addr, val, oi, ra);
}

void helper_hyp_hsv_d(CPURISCVState *env, target_ulong addr, target_ulong val)
{
    uintptr_t ra = GETPC();
    int mmu_idx = check_access_hlsv(env, false, ra);
    MemOpIdx oi = make_memop_idx(MO_TEUQ, mmu_idx);

    cpu_stq_mmu(env, addr, val, oi, ra);
}

/*
 * TODO: These implementations are not quite correct.  They perform the
 * access using execute permission just fine, but the final PMP check
 * is supposed to have read permission as well.  Without replicating
 * a fair fraction of cputlb.c, fixing this requires adding new mmu_idx
 * which would imply that exact check in tlb_fill.
 */
target_ulong helper_hyp_hlvx_hu(CPURISCVState *env, target_ulong addr)
{
    uintptr_t ra = GETPC();
    int mmu_idx = check_access_hlsv(env, true, ra);
    MemOpIdx oi = make_memop_idx(MO_TEUW, mmu_idx);

    return cpu_ldw_code_mmu(env, addr, oi, GETPC());
}

target_ulong helper_hyp_hlvx_wu(CPURISCVState *env, target_ulong addr)
{
    uintptr_t ra = GETPC();
    int mmu_idx = check_access_hlsv(env, true, ra);
    MemOpIdx oi = make_memop_idx(MO_TEUL, mmu_idx);

    return cpu_ldl_code_mmu(env, addr, oi, ra);
}

#endif /* !CONFIG_USER_ONLY */

target_ulong helper_mulah(target_ulong src1, target_ulong src2,
                          target_ulong dst)
{
    int16_t a, b;
    int32_t c, d;
    a = (int16_t)(src1);
    b = (int16_t)(src2);
    c = (int32_t)dst;
    d = (int32_t)a * (int32_t)b;
    d = c + d;
    return (target_long)d;
}

target_ulong helper_mulsh(target_ulong src1, target_ulong src2,
                          target_ulong dst)
{
    int16_t a, b;
    int32_t c, d;
    a = (int16_t)(src1);
    b = (int16_t)(src2);
    c = (int32_t)dst;
    d = (int32_t)a * (int32_t)b;
    d = c - d;
    return (target_long)d;
}

target_ulong helper_tstnbz(target_ulong src, target_ulong ol)
{
    int i;
    target_ulong res = 0;
    int cnt = ol == MXL_RV32 ? 4 : sizeof(target_ulong);
    for (i = 0; i < cnt; i++) {
        if ((src & 0xff) == 0) {
            res |= ((target_ulong)0xff << (i * 8));
        }
        src = src >> 8;
    }
    return res;
}

target_ulong helper_ff0(target_ulong src, target_ulong ol)
{
    int i = 0;
    int bits = ol == MXL_RV32 ? 32 : sizeof(target_ulong) * 8;

    for (i = 0; i < bits; i++) {
        if ((src & ((target_ulong)1 << (bits - 1 - i))) == 0) {
            break;
        }
    }
    return i;
}

target_ulong helper_ff1(target_ulong src, target_ulong ol)
{
    int i = 0;
    int bits =  ol == MXL_RV32 ? 32 : sizeof(target_ulong) * 8;

    for (i = 0; i < bits; i++) {
        if ((src & ((target_ulong)1 << (bits - 1 - i))) != 0) {
            break;
        }
    }
    return i;
}

target_ulong helper_srri(target_ulong src, target_ulong imm, target_ulong ol)
{
    target_ulong i = 0;
    int bits =  ol == MXL_RV32 ? 32 : sizeof(target_ulong) * 8;

    i = (src >> imm) | (src << (bits - imm));
    return i;
}

target_ulong helper_tst(target_ulong src, target_ulong imm)
{
    target_ulong i = 0;

    if (src & (1ULL << imm)) {
        i = 1;
    }

    return i;
}

target_ulong helper_srriw(target_ulong src, target_ulong imm)
{
    target_ulong i = 0;
    uint32_t tmp = src & 0xffffffff;
    tmp = ((tmp >> imm) | (tmp << (32 - imm)));
    if (tmp & 0x80000000) {
        i = 0xffffffff00000000 | tmp;
    } else {
        i = tmp;
    }
    return i;
}

#ifndef CONFIG_USER_ONLY
void helper_ipush(CPURISCVState *env)
{
    target_ulong base = env->gpr[2];
    int i = 4;
    /* TODO: probe the memory */
    for(; i <= 72; i += 4) {
        switch (i) {
        case 4:
            cpu_stl_data(env, base - i, env->mcause);
            break;
        case 8:
            cpu_stl_data(env, base - i, env->mepc);
            break;
        case 12: /* X1 */
            cpu_stl_data(env, base - i, env->gpr[1]);
            break;
        case 16: /* X5-X7 */
        case 20:
        case 24:
            cpu_stl_data(env, base - i, env->gpr[i / 4 + 1]);
            break;
        case 28: /* X10-X17 */
        case 32:
        case 36:
        case 40:
        case 44:
        case 48:
        case 52:
        case 56:
            cpu_stl_data(env, base - i, env->gpr[i / 4 + 3]);
            break;
        case 60: /* X28-X31 */
        case 64:
        case 68:
        case 72:
            cpu_stl_data(env, base - i, env->gpr[i / 4 + 13]);
            break;
        }
    }
    env->gpr[2] -= 72;
    env->mstatus = set_field(env->mstatus, MSTATUS_MIE, 1);
}

target_ulong helper_ipop(CPURISCVState *env, target_ulong cpu_pc_deb)
{
    target_ulong base = env->gpr[2];
    int i = 68;
    /* TODO: probe the memory */
    for(; i >= 0; i -= 4) {
        switch (i) {
        case 0: /* X31-X28 */
        case 4:
        case 8:
        case 12:
            env->gpr[31 - i / 4] = cpu_ldl_data(env, base + i);
            break;
        case 16: /* X17-X10 */
        case 20:
        case 24:
        case 28:
        case 32:
        case 36:
        case 40:
        case 44:
            env->gpr[21 - i / 4] = cpu_ldl_data(env, base + i);
            break;
        case 48: /* X7-X5 */
        case 52:
        case 56:
            env->gpr[19 - i / 4] = cpu_ldl_data(env, base + i);
            break;
        case 60: /* X1 */
            env->gpr[1] = cpu_ldl_data(env, base + i);
            break;
        case 64:
            env->mepc = cpu_ldl_data(env, base + i);
            break;
        case 68:
            env->mcause = cpu_ldl_data(env, base + i);
            break;
        }
    }
    env->gpr[2] += 72;
    return do_excp_return(env, GETPC());
}
#endif
