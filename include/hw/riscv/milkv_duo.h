/*
 * Milkv DUO machine interface
 *
 * Copyright (c) 2023 PLCT Lab.
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

#ifndef HW_MILKV_DUO_H
#define HW_MILKV_DUO_H

#include "hw/riscv/riscv_hart.h"
#include "hw/boards.h"
#include "hw/timer/milkv_timer.h"

#define TYPE_RISCV_DUO_SOC "riscv.milkv.duo.soc"
#define RISCV_DUO_SOC(obj) \
    OBJECT_CHECK(MilkvDuoSoCState, (obj), TYPE_RISCV_DUO_SOC)

typedef struct MilkvDuoSoCState {
    /*< private >*/
    DeviceState parent_obj;

    /*< public >*/
    RISCVHartArrayState cpus;

    DuoTimerState timer;

    DeviceState *plic;
} MilkvDuoSoCState;

typedef struct MilkvDuoState {
    /*< private >*/
    MachineState parent_obj;

    /*< public >*/
    MilkvDuoSoCState soc;
    int fdt_size;
} MilkvDuoState;

#define TYPE_RISCV_DUO_MACHINE MACHINE_TYPE_NAME("milkv_duo")
#define RISCV_DUO_MACHINE(obj) \
    OBJECT_CHECK(MilkvDuoState, (obj), TYPE_RISCV_DUO_MACHINE)

enum {
    MILKV_DUO_DEV_MAILBOX,
    MILKV_DUO_DEV_SYSCTRL,
    MILKV_DUO_DEV_CLINT,
    MILKV_DUO_DEV_TOP_MISC,
    MILKV_DUO_DEV_PINMUX,
    MILKV_DUO_DEV_PLL,
    MILKV_DUO_DEV_RSTGEN,
    MILKV_DUO_DEV_WDT0,
    MILKV_DUO_DEV_WDT1,
    MILKV_DUO_DEV_WDT2,
    MILKV_DUO_DEV_GPIO0,
    MILKV_DUO_DEV_GPIO1,
    MILKV_DUO_DEV_GPIO2,
    MILKV_DUO_DEV_GPIO3,
    MILKV_DUO_DEV_PWM0,
    MILKV_DUO_DEV_PWM1,
    MILKV_DUO_DEV_PWM2,
    MILKV_DUO_DEV_PWM3,
    MILKV_DUO_DEV_TIMER,
    MILKV_DUO_DEV_I2C0,
    MILKV_DUO_DEV_I2C1,
    MILKV_DUO_DEV_I2C2,
    MILKV_DUO_DEV_I2C3,
    MILKV_DUO_DEV_I2C4,
    MILKV_DUO_DEV_UART0,
    MILKV_DUO_DEV_UART1,
    MILKV_DUO_DEV_UART2,
    MILKV_DUO_DEV_UART3,
    MILKV_DUO_DEV_UART4,
    MILKV_DUO_DEV_SPI0,
    MILKV_DUO_DEV_SPI1,
    MILKV_DUO_DEV_SPI2,
    MILKV_DUO_DEV_SPI3,
    MILKV_DUO_DEV_PLIC,
    MILKV_DUO_DEV_DDR
};

#define MILKV_DUO_PLIC_HART_CONFIG "M"

enum {
    MILKV_DUO_UART0_IRQ   = 44,
    MILKV_DUO_UART1_IRQ   = 45,
    MILKV_DUO_UART2_IRQ   = 46,
    MILKV_DUO_UART3_IRQ   = 47,
    MILKV_DUO_UART4_IRQ   = 48,
    MILKV_DUO_TIMER0_IRQ  = 79,
    MILKV_DUO_TIMER1_IRQ  = 80,
    MILKV_DUO_TIMER2_IRQ  = 81,
    MILKV_DUO_TIMER3_IRQ  = 82,
    MILKV_DUO_TIMER4_IRQ  = 83,
    MILKV_DUO_TIMER5_IRQ  = 84,
    MILKV_DUO_TIMER6_IRQ  = 85,
    MILKV_DUO_TIMER7_IRQ  = 86,
};

#endif
