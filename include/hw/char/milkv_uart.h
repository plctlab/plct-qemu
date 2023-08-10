/*
 * Milkv DUO UART interface
 *
 * Copyright (c) 2023 PLCT Lab
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

#ifndef HW_MILKV_UART_H
#define HW_MILKV_UART_H

#include "chardev/char-fe.h"
#include "hw/qdev-properties.h"
#include "hw/sysbus.h"
#include "qom/object.h"

enum {
    DUO_UART_RBR_THR_DLL   = 0,
    DUO_UART_IER_DLH       = 4,
    DUO_UART_FCR_IIR       = 8,
    DUO_UART_LCR           = 12,
    DUO_UART_MCR           = 16,
    DUO_UART_LSR           = 20,
    DUO_UART_MSR           = 24,
    DUO_UART_LPDLL         = 32,
    DUO_UART_LPDLH         = 36,
    DUO_UART_SRBR_STHR     = 48,
    DUO_UART_FAR           = 112,
    DUO_UART_TFR           = 116,
    DUO_UART_RFW           = 120,
    DUO_UART_USR           = 124,
    DUO_UART_TFL           = 128,
    DUO_UART_RFL           = 132,
    DUO_UART_SRR           = 136,
    DUO_UART_SRTS          = 140,
    DUO_UART_SBCR          = 144,
    DUO_UART_SDMAM         = 148,
    DUO_UART_SFE           = 152,
    DUO_UART_SRT           = 156,
    DUO_UART_STET          = 160,
    DUO_UART_HTX           = 164,
    DUO_UART_DMASA         = 168,
    DUO_UART_MAX           = 172
};

enum {
    DUO_UART_IE_RX_AVAIL        = 0,
    DUO_UART_IE_TX_EMPTY        = 1,
    DUO_UART_IE_RX_LINE_STATUS  = 2,
    DUO_UART_IE_MODEM_STATUS    = 3,
};

enum {
    DUO_IRQ_MODEM_STATUS        = 0,
    DUO_IRQ_NO_PENDING          = 1,
    DUO_IRQ_THR_EMPTY           = 2,
    DUO_IRQ_RX_AVAIL            = 4,
    DUO_IRQ_THR_LINE_STATUS     = 6,
    DUO_IRQ_BUSY                = 7,
    DUO_IRQ_CHAR_TIMEOUT        = 12
};

#define DUO_UART_RX_FIFO_SIZE 64

#define TYPE_DUO_UART "riscv.duo.uart"
OBJECT_DECLARE_SIMPLE_TYPE(DuoUARTState, DUO_UART)

struct DuoUARTState {
    /*< private >*/
    SysBusDevice parent_obj;

    /*< public >*/
    qemu_irq irq;
    MemoryRegion mmio;
    CharBackend chr;
    uint8_t rx_fifo[DUO_UART_RX_FIFO_SIZE];
    uint8_t rx_fifo_len;
    uint8_t ie;
    uint16_t div;
    uint8_t lcr;
    uint8_t irq_id;
    uint8_t fcr;
    uint8_t mcr;
    uint8_t lsr;
    uint16_t lpdiv;
    uint8_t usr;
};

#define LCR_DIVISOR_ACCESS_BIT  (1UL << 7)

#define IER_RX_READY            (1UL << 0)
#define IER_THR_EMPTY           (1UL << 1)
#define IER_THR_RX_LINE_STATUS  (1UL << 2)
#define IER_THR_MODEM_STATUS    (1UL << 3)
#define IER_THRE_INTMODE_ENABLE (1UL << 7)

#define FCR_FIFO_ENABLE         (1UL << 0)
#define FCR_FIFO_RX_ET          (1UL << 1)
#define FCR_FIFO_TX_RESET       (1UL << 2)
#define FCR_FIFO_DMA_MODE       (1UL << 3)
#define FCR_FIFO_TX_EMPTY(x)    ((x >> 4) & 0x3)
#define FCR_FIFO_RX_FULL(x)     ((x >> 6) & 0x3)

enum {
    TX_FIFO_EMPTY,
    TX_TWO_CHARS,
    TX_QUAT_FIFO,
    TX_HALF_FIFO
};

enum {
    RX_ONE_CHAR,
    RX_QUAT_FIFO,
    RX_HALF_FIFO,
    RX_2LESS_FULL_FIFO
};

#define LSR_RX_DATA_READY       (1UL << 0)
#define LSR_TX_THR_EMPTY        (5UL << 1)
#define LSR_TX_EMPTY            (6UL << 1)
#define LSR_RX_FIFO_ERROR       (7UL << 0)

#define USR_UART_BUSY           (1UL << 0)
#define USR_UART_TX_NOT_FULL    (1UL << 1)
#define USR_UART_TX_EMPTY       (1UL << 2)
#define USR_UART_RX_NOT_EMPTY   (1UL << 3)
#define USR_UART_RX_FULL        (1UL << 4)

#define DUO_UART_MAX_NUM        5
DuoUARTState *duo_uart_create(MemoryRegion *address_space, hwaddr base,
                              Chardev *chr, qemu_irq irq);

#endif
