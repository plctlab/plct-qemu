/*
 * QEMU RISC-V Board Compatible with Milkv DUO SDK
 *
 * Copyright (c) 2023 PLCT Lab.
 *
 * Provides a board compatible with the Milkv DUO
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
#include "qemu/cutils.h"
#include "qemu/error-report.h"
#include "qapi/error.h"
#include "hw/boards.h"
#include "hw/loader.h"
#include "hw/sysbus.h"
#include "hw/char/serial.h"
#include "hw/misc/unimp.h"
#include "target/riscv/cpu.h"
#include "hw/riscv/riscv_hart.h"
#include "hw/riscv/milkv_duo.h"
#include "hw/riscv/boot.h"
#include "chardev/char.h"
#include "sysemu/sysemu.h"
#include "hw/intc/sifive_plic.h"
#include "hw/intc/riscv_aclint.h"
#include "hw/riscv/virt.h"
#include "hw/char/milkv_uart.h"

static const MemMapEntry milkv_duo_memmap[] = {
    [MILKV_DUO_DEV_MAILBOX]  =     { 0x01900000,    0x1000 },
    [MILKV_DUO_DEV_SYSCTRL]  =     { 0x01901000,    0x1000 },
    [MILKV_DUO_DEV_TOP_MISC] =     { 0x03000000,    0x1000 },
    [MILKV_DUO_DEV_PINMUX]   =     { 0x03001000,    0x1000 },
    [MILKV_DUO_DEV_PLL]      =     { 0x03002000,    0x1000 },
    [MILKV_DUO_DEV_RSTGEN]   =     { 0x03003000,    0x1000 },
    [MILKV_DUO_DEV_WDT0]     =     { 0x03010000,    0x1000 },
    [MILKV_DUO_DEV_WDT1]     =     { 0x03011000,    0x1000 },
    [MILKV_DUO_DEV_WDT2]     =     { 0x03012000,    0x1000 },
    [MILKV_DUO_DEV_GPIO0]    =     { 0x03020000,    0x1000 },
    [MILKV_DUO_DEV_GPIO1]    =     { 0x03021000,    0x1000 },
    [MILKV_DUO_DEV_GPIO2]    =     { 0x03022000,    0x1000 },
    [MILKV_DUO_DEV_GPIO3]    =     { 0x03023000,    0x1000 },
    [MILKV_DUO_DEV_PWM0]     =     { 0x03060000,    0x1000 },
    [MILKV_DUO_DEV_PWM1]     =     { 0x03061000,    0x1000 },
    [MILKV_DUO_DEV_PWM2]     =     { 0x03062000,    0x1000 },
    [MILKV_DUO_DEV_PWM3]     =     { 0x03063000,    0x1000 },
    [MILKV_DUO_DEV_TIMER]    =     { 0x030A0000,   0x10000 },
    [MILKV_DUO_DEV_I2C0]     =     { 0x04000000,   0x10000 },
    [MILKV_DUO_DEV_I2C1]     =     { 0x04010000,   0x10000 },
    [MILKV_DUO_DEV_I2C2]     =     { 0x04020000,   0x10000 },
    [MILKV_DUO_DEV_I2C3]     =     { 0x04030000,   0x10000 },
    [MILKV_DUO_DEV_I2C4]     =     { 0x04040000,   0x10000 },
    [MILKV_DUO_DEV_UART0]    =     { 0x04140000,   0x10000 },
    [MILKV_DUO_DEV_UART1]    =     { 0x04150000,   0x10000 },
    [MILKV_DUO_DEV_UART2]    =     { 0x04160000,   0x10000 },
    [MILKV_DUO_DEV_UART3]    =     { 0x04170000,   0x10000 },
    [MILKV_DUO_DEV_SPI0]     =     { 0x04180000,   0x10000 },
    [MILKV_DUO_DEV_SPI1]     =     { 0x04190000,   0x10000 },
    [MILKV_DUO_DEV_SPI2]     =     { 0x041A0000,   0x10000 },
    [MILKV_DUO_DEV_SPI3]     =     { 0x041B0000,   0x10000 },
    [MILKV_DUO_DEV_UART4]    =     { 0x041C0000,   0x10000 },
    [MILKV_DUO_DEV_PLIC]     =     { 0x70000000, 0x4000000 },
    [MILKV_DUO_DEV_CLINT]    =     { 0x74000000,   0x10000 },
    [MILKV_DUO_DEV_DDR]      =     { 0x80000000,         0 },
};

static void milkv_duo_machine_init(MachineState *machine)
{
    MachineClass *mc = MACHINE_GET_CLASS(machine);
    const MemMapEntry *memmap = milkv_duo_memmap;

    MilkvDuoState *s = RISCV_DUO_MACHINE(machine);
    MemoryRegion *sys_mem = get_system_memory();

    if (machine->ram_size != mc->default_ram_size) {
        char *sz = size_to_str(mc->default_ram_size);
        error_report("Invalid RAM size, should be %s", sz);
        g_free(sz);
        exit(EXIT_FAILURE);
    }

    /* Initialize SoC */
    object_initialize_child(OBJECT(machine), "soc", &s->soc, TYPE_RISCV_DUO_SOC);
    qdev_realize(DEVICE(&s->soc), NULL, &error_fatal);

    /* Data Tightly Integrated Memory */
    memory_region_add_subregion(sys_mem,
        memmap[MILKV_DUO_DEV_DDR].base, machine->ram);

    if (machine->kernel_filename) {
        riscv_load_kernel(machine, &s->soc.cpus,
                          memmap[MILKV_DUO_DEV_DDR].base,
                          false, NULL);
    }
}

static void milkv_duo_machine_instance_init(Object *obj)
{
    // MilkvDuoState *s = RISCV_DUO_MACHINE(obj);
}

static void milkv_duo_machine_class_init(ObjectClass *oc, void *data)
{
    MachineClass *mc = MACHINE_CLASS(oc);

    mc->desc = "RISC-V Board compatible with Milkv DUO SDK";
    mc->init = milkv_duo_machine_init;
    mc->max_cpus = 1;
    mc->default_cpu_type = RISCV_CPU_TYPE_NAME("c906fdv");;
    mc->default_ram_id = "riscv.milkv.duo.ram";
}

static const TypeInfo milkv_duo_machine_typeinfo = {
    .name       = MACHINE_TYPE_NAME("milkv_duo"),
    .parent     = TYPE_MACHINE,
    .class_init = milkv_duo_machine_class_init,
    .instance_init = milkv_duo_machine_instance_init,
    .instance_size = sizeof(MilkvDuoState),
};

static void milkv_duo_machine_init_register_types(void)
{
    type_register_static(&milkv_duo_machine_typeinfo);
}

type_init(milkv_duo_machine_init_register_types)

static void milkv_duo_soc_init(Object *obj)
{
    MachineState *ms = MACHINE(qdev_get_machine());
    MilkvDuoSoCState *s = RISCV_DUO_SOC(obj);

    object_initialize_child(obj, "cpus", &s->cpus, TYPE_RISCV_HART_ARRAY);
    object_property_set_int(OBJECT(&s->cpus), "num-harts", ms->smp.cpus,
                            &error_abort);

    object_property_set_int(OBJECT(&s->cpus), "resetvec", 0x80200000, &error_abort);

    object_initialize_child(obj, "timer", &s->timer, TYPE_DUO_TIMER);
}

static void milkv_duo_soc_realize(DeviceState *dev, Error **errp)
{
    MachineState *ms = MACHINE(qdev_get_machine());
    const MemMapEntry *memmap = milkv_duo_memmap;
    MilkvDuoSoCState *s = RISCV_DUO_SOC(dev);
    int j;

    object_property_set_str(OBJECT(&s->cpus), "cpu-type", ms->cpu_type,
                            &error_abort);
        
    sysbus_realize(SYS_BUS_DEVICE(&s->cpus), &error_fatal);

    
    s->plic = sifive_plic_create(memmap[MILKV_DUO_DEV_PLIC].base,
        (char *)MILKV_DUO_PLIC_HART_CONFIG, ms->smp.cpus, 0,
        VIRT_IRQCHIP_NUM_SOURCES,
        ((1U << VIRT_IRQCHIP_NUM_PRIO_BITS) - 1),
        VIRT_PLIC_PRIORITY_BASE,
        VIRT_PLIC_PENDING_BASE,
        VIRT_PLIC_ENABLE_BASE,
        VIRT_PLIC_ENABLE_STRIDE,
        VIRT_PLIC_CONTEXT_BASE,
        VIRT_PLIC_CONTEXT_STRIDE,
        memmap[MILKV_DUO_DEV_PLIC].size);
    riscv_aclint_swi_create(memmap[MILKV_DUO_DEV_CLINT].base, 0,
                            ms->smp.cpus, false);
    riscv_aclint_mtimer_create(memmap[MILKV_DUO_DEV_CLINT].base +
            RISCV_ACLINT_SWI_SIZE,
        RISCV_ACLINT_DEFAULT_MTIMER_SIZE, 0, ms->smp.cpus,
        RISCV_ACLINT_DEFAULT_MTIMECMP, RISCV_ACLINT_DEFAULT_MTIME,
        RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ, true);

    /* TIMER */
    if (!sysbus_realize(SYS_BUS_DEVICE(&s->timer), errp)) {
        return;
    }
    sysbus_mmio_map(SYS_BUS_DEVICE(&s->timer), 0,
                    memmap[MILKV_DUO_DEV_TIMER].base);

    /* Connect TIMER interrupts to the PLIC */
    for (j = 0; j < DUO_TIMER_MAX_NUM; j++) {
        sysbus_connect_irq(SYS_BUS_DEVICE(&s->timer), j,
                           qdev_get_gpio_in(DEVICE(s->plic),
                                            MILKV_DUO_TIMER0_IRQ + j));
    }

    for (j = 0; j < DUO_UART_MAX_NUM; j++) {
        duo_uart_create(get_system_memory(),
                        memmap[MILKV_DUO_DEV_UART0 + j].base, serial_hd(j),
                        qdev_get_gpio_in(DEVICE(s->plic),
                                         MILKV_DUO_UART0_IRQ + j));
    }
}

static void milkv_duo_soc_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);

    dc->realize = milkv_duo_soc_realize;
    /* Reason: Uses serial_hds in realize function, thus can't be used twice */
    dc->user_creatable = false;
}

static const TypeInfo milkv_duo_soc_type_info = {
    .name = TYPE_RISCV_DUO_SOC,
    .parent = TYPE_DEVICE,
    .instance_size = sizeof(MilkvDuoSoCState),
    .instance_init = milkv_duo_soc_init,
    .class_init = milkv_duo_soc_class_init,
};

static void milkv_duo_soc_register_types(void)
{
    type_register_static(&milkv_duo_soc_type_info);
}

type_init(milkv_duo_soc_register_types)
