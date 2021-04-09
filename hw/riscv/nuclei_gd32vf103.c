/*
 * QEMU RISC-V Nuclei  GD32VF103 Board Compatible  with Nuclei SDK
 *
 * Copyright (c) 2020  PLCT Lab
 * Copyright (c) 2020 Gao ZhiYuan <alapha23@gmail.com>
 *
 * Provides a board compatible with the Nuclei SDK:
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
#include "qemu/log.h"
#include "qemu/error-report.h"
#include "qapi/error.h"
#include "hw/boards.h"
#include "hw/loader.h"
#include "hw/sysbus.h"
#include "target/riscv/cpu.h"
#include "hw/misc/unimp.h"
#include "hw/char/riscv_htif.h"
#include "hw/riscv/riscv_hart.h"
#include "hw/intc/nuclei_eclic.h"
#include "hw/char/gd32vf103_usart.h"
#include "hw/riscv/nuclei_gd32vf103.h"
#include "hw/riscv/boot.h"
#include "chardev/char.h"
#include "sysemu/arch_init.h"
#include "sysemu/device_tree.h"
#include "sysemu/qtest.h"
#include "sysemu/sysemu.h"
#include "exec/address-spaces.h"

#include <libfdt.h>

static const struct MemmapEntry {
    hwaddr base;
    hwaddr size;
} gd32vf103_memmap[] = {
    [GD32VF103_EXMC_SWREG] = { 0xA0000000,     0x1000 },
    [GD32VF103_EXMC_SWREG] = { 0x60000000,     0x10000000 },
    [GD32VF103_USBFS]      = { 0x50000000,     0x100000 },
    [GD32VF103_CRC]        = { 0x40023000,     0x400 },
    [GD32VF103_FMC]        = { 0x40022000,     0x400 },
    [GD32VF103_RCU]        = { 0x40021000,     0x400 },
    [GD32VF103_DMA1]       = { 0x40020400,     0x400 },
    [GD32VF103_DMA0]       = { 0x40020000,     0x400 },
    [GD32VF103_UART4]     = { 0x40005000,     0x400 },
    [GD32VF103_USART0]     = { 0x40013800,     0x400 },
    [GD32VF103_SRAM]       = { 0x20000000,     0x18000 },
    [GD32VF103_OB]         = { 0x1FFFF800,     0x10 },
    [GD32VF103_BL]         = { 0x1FFFB000,     0x800 },
    [GD32VF103_MAINFLASH]  = { 0x8000000,     0x20000 },
    [GD32VF103_MFOL]       = { 0x0,     0x20000 },
    [GD32VF103_ECLIC]      = { 0xD2000000,     0x10000 },
    [GD32VF103_TIMER0]      = { 0x40012C00,     0x400 },
    [GD32VF103_SYSTIMER]      = { 0xD1000000,     0x1000 },
    [GD32VF103_AFIO]      = { 0x40010000,     0x400 },
    [GD32VF103_GPIOA]      = { 0x40010800,     0x400 },
    [GD32VF103_GPIOB]      = { 0x40010C00,     0x400 },
    [GD32VF103_GPIOC]      = { 0x40011000,     0x400 },
    [GD32VF103_GPIOD]      = { 0x40011400,     0x400 },
    [GD32VF103_GPIOE]      = { 0x40011800,     0x400 },
};

static void nuclei_board_init(MachineState *machine)
{
    const struct MemmapEntry *memmap = gd32vf103_memmap;
    NucleiGDState *s = g_new0(NucleiGDState, 1);
    MemoryRegion *system_memory = get_system_memory();
    MemoryRegion *main_mem = g_new(MemoryRegion, 1);
    s->soc.sram = *main_mem;
    int i;

    /* TODO: Add qtest support */
    /* Initialize SOC */
    object_initialize_child(OBJECT(machine), "soc", &s->soc, TYPE_NUCLEI_GD32VF103_SOC);
    qdev_realize(DEVICE(&s->soc), NULL, &error_abort);

    memory_region_init_ram(&s->soc.main_flash, NULL, "riscv.nuclei.main_flash",
                           memmap[GD32VF103_MAINFLASH].size, &error_fatal);
    memory_region_add_subregion(system_memory,
    memmap[GD32VF103_MAINFLASH].base, &s->soc.main_flash);

        memory_region_init_ram(&s->soc.boot_loader, NULL, "riscv.nuclei.boot_loader",
                           memmap[GD32VF103_BL].size, &error_fatal);
    memory_region_add_subregion(system_memory,
    memmap[GD32VF103_BL].base, &s->soc.boot_loader);

        memory_region_init_ram(&s->soc.ob, NULL, "riscv.nuclei.ob",
                           memmap[GD32VF103_OB].size, &error_fatal);
    memory_region_add_subregion(system_memory,
                    memmap[GD32VF103_OB].base, &s->soc.ob);

    memory_region_init_ram(&s->soc.sram, NULL, "riscv.nuclei.sram",
                           memmap[GD32VF103_SRAM].size, &error_fatal);
    memory_region_add_subregion(system_memory,
                    memmap[GD32VF103_SRAM].base, &s->soc.sram);

    /* reset vector */
    uint32_t reset_vec[8] = {
        0x00000297,                    /* 1:  auipc  t0, %pcrel_hi(dtb) */
        0x02028593,                    /*     addi   a1, t0, %pcrel_lo(1b) */
        0xf1402573,                    /*     csrr   a0, mhartid  */
#if defined(TARGET_RISCV32)
        0x0182a283,                    /*     lw     t0, 24(t0) */
#elif defined(TARGET_RISCV64)
        0x0182b283,                    /*     ld     t0, 24(t0) */
#endif
        0x00028067,                    /*     jr     t0 */
        0x00000000,
        memmap[GD32VF103_MAINFLASH].base, /* start: .dword */
        0x00000000,
                                       /* dtb: */
    };

    /* copy in the reset vector in little_endian byte order */
    for (i = 0; i < sizeof(reset_vec) >> 2; i++) {
        reset_vec[i] = cpu_to_le32(reset_vec[i]);
    }
    rom_add_blob_fixed_as("mrom.reset", reset_vec, sizeof(reset_vec),
                          memmap[GD32VF103_MFOL].base + 0x1000, &address_space_memory);

    /* boot rom */
    if (machine->kernel_filename) {
        riscv_load_kernel(machine->kernel_filename,
            memmap[GD32VF103_MAINFLASH].base, NULL);
    }
}

static void riscv_nuclei_soc_init(Object *obj)
{
    MachineState *ms = MACHINE(qdev_get_machine());
    NucleiGDSoCState *s = RISCV_NUCLEI_GD32VF103_SOC(obj);

    object_initialize_child(obj, "cpus", &s->cpus, TYPE_RISCV_HART_ARRAY);

    object_property_set_int(OBJECT(&s->cpus), "num-harts", ms->smp.cpus,
                            &error_abort);

    object_initialize_child(obj, "timer",
                          &s->timer,
                          TYPE_GD32VF103_TIMER);

    // object_initialize_child(obj, "systimer",
    //                       &s->systimer,
    //                       TYPE_NUCLEI_SYSTIMER);

    object_initialize_child(obj, "rcu",
                          &s->rcu,
                          TYPE_NUCLEI_RCU);

    object_initialize_child(obj, "gpioa",
                          &s->gpioa,
                          TYPE_GD32VF103_GPIO);

    object_initialize_child(obj, "gpiob",
                          &s->gpiob,
                          TYPE_GD32VF103_GPIO);

    object_initialize_child(obj, "gpioc",
                          &s->gpioc,
                          TYPE_GD32VF103_GPIO);

    object_initialize_child(obj, "gpiod",
                          &s->gpiod,
                          TYPE_GD32VF103_GPIO);

    object_initialize_child(obj, "gpioe",
                          &s->gpioe,
                          TYPE_GD32VF103_GPIO);
}

static void riscv_nuclei_soc_realize(DeviceState *dev, Error **errp)
{
    MachineState *ms = MACHINE(qdev_get_machine());
    const struct MemmapEntry *memmap = gd32vf103_memmap;
    NucleiGDSoCState *s = RISCV_NUCLEI_GD32VF103_SOC(dev);
    MemoryRegion *sys_mem = get_system_memory();

    Error *err = NULL;

    object_property_set_str(OBJECT(&s->cpus), "cpu-type",ms->cpu_type,
                            &error_abort);
    sysbus_realize(SYS_BUS_DEVICE(&s->cpus), &error_abort);

    /* Mask ROM */
    memory_region_init_rom(&s->internal_rom, OBJECT(dev), "riscv.nuclei.irom",
                           memmap[GD32VF103_MFOL].size, &error_fatal);
    memory_region_add_subregion(sys_mem,
        memmap[GD32VF103_MFOL].base, &s->internal_rom);

    /* MMIO */
     s->eclic = nuclei_eclic_create(memmap[GD32VF103_ECLIC].base,
         memmap[GD32VF103_ECLIC].size, GD32VF103_SOC_INT_MAX);

    nuclei_systimer_create(memmap[GD32VF103_SYSTIMER].base,
                memmap[GD32VF103_SYSTIMER].size,
                 s->eclic,
                NUCLEI_GD32_TIMEBASE_FREQ);

    /* Timer*/
    sysbus_realize(SYS_BUS_DEVICE(&s->timer), &err);
    if (err) {
        error_propagate(errp, err);
        return;
    }
    sysbus_mmio_map(SYS_BUS_DEVICE(&s->timer), 0, memmap[GD32VF103_TIMER0].base);

    /* RCU*/
    sysbus_realize(SYS_BUS_DEVICE(&s->rcu), &err);
    if (err) {
        error_propagate(errp, err);
        return;
    }
    sysbus_mmio_map(SYS_BUS_DEVICE(&s->rcu), 0, memmap[GD32VF103_RCU].base);

    /* GPIO */
    sysbus_realize(SYS_BUS_DEVICE(&s->gpioa), &err);
    sysbus_realize(SYS_BUS_DEVICE(&s->gpiob), &err);
    sysbus_realize(SYS_BUS_DEVICE(&s->gpioc), &err);
    sysbus_realize(SYS_BUS_DEVICE(&s->gpiod), &err);
    sysbus_realize(SYS_BUS_DEVICE(&s->gpioe), &err);
    if (err) {
        error_propagate(errp, err);
        return;
    }

    // /* Map GPIO registers */
    sysbus_mmio_map(SYS_BUS_DEVICE(&s->gpioa), 0, memmap[GD32VF103_GPIOA].base);
    sysbus_mmio_map(SYS_BUS_DEVICE(&s->gpiob), 0, memmap[GD32VF103_GPIOB].base);
    sysbus_mmio_map(SYS_BUS_DEVICE(&s->gpioc), 0, memmap[GD32VF103_GPIOC].base);
    sysbus_mmio_map(SYS_BUS_DEVICE(&s->gpiod), 0, memmap[GD32VF103_GPIOD].base);
    sysbus_mmio_map(SYS_BUS_DEVICE(&s->gpioe), 0, memmap[GD32VF103_GPIOE].base);
    /* Pass all GPIOs to the SOC layer so they are available to the board */

    /* Create and connect USART interrupts to the ECLIC */
    gd32vf103_usart_create(sys_mem,
                    memmap[GD32VF103_UART4].base,
                    memmap[GD32VF103_UART4].size,
                    serial_hd(0),
                    nuclei_eclic_get_irq(DEVICE(s->eclic),
                    GD32VF103_UART4_IRQn));



    /* Flash memory */
    // memory_region_init_rom(&s->xip_mem, OBJECT(dev), "riscv.nuclei.xip",
    //                     memmap[GD32VF103_MAINFLASH].size, &error_fatal);
    // memory_region_add_subregion(sys_mem,
    //                     memmap[GD32VF103_MAINFLASH].base, &s->xip_mem);
}

static void nuclei_eval_machine_init(MachineClass *mc)
{
    mc->desc = "RISC-V Nuclei GD32VF103 Eval Board";
    mc->init = nuclei_board_init;
    mc->max_cpus = 1;
    mc->is_default = false;
    mc->default_cpu_type = NUCLEI_CPU;
}

static void nuclei_rvstar_machine_init(MachineClass *mc)
{
    mc->desc = "RISC-V Nuclei GD32VF103 Rvstar Board";
    mc->init = nuclei_board_init;
    mc->max_cpus = 1;
    mc->is_default = false;
    mc->default_cpu_type = NUCLEI_CPU;
}

DEFINE_MACHINE("gd32vf103_eval", nuclei_eval_machine_init)
DEFINE_MACHINE("gd32vf103_rvstar", nuclei_rvstar_machine_init)

static void riscv_nuclei_soc_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    dc->realize = riscv_nuclei_soc_realize;
    dc->user_creatable = false;
}

static const TypeInfo nuclei_gd_soc_type_info = {
    .name = TYPE_NUCLEI_GD32VF103_SOC,
    .parent = TYPE_DEVICE,
    .instance_size = sizeof(NucleiGDSoCState),
    .instance_init = riscv_nuclei_soc_init,
    .class_init = riscv_nuclei_soc_class_init,
};

static void nuclei_gd_soc_register_types(void)
{
    type_register_static(&nuclei_gd_soc_type_info);
}

type_init(nuclei_gd_soc_register_types)