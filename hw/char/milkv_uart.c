/*
 * QEMU model of the UART on the Milkv DUO Board.
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

#include "qemu/osdep.h"
#include "qapi/error.h"
#include "qemu/log.h"
#include "migration/vmstate.h"
#include "chardev/char.h"
#include "chardev/char-fe.h"
#include "hw/irq.h"
#include "hw/char/milkv_uart.h"
#include "hw/qdev-properties-system.h"


static void duo_uart_update_status(DuoUARTState *s)
{
    bool cond = false;
    bool rx_full = false;

    s->lsr |= LSR_TX_EMPTY | LSR_TX_THR_EMPTY;

    if (s->fcr & FCR_FIFO_ENABLE) {
        s->usr = USR_UART_TX_EMPTY | USR_UART_TX_NOT_FULL;
    } else {
        s->usr = 0;
    }

    if (s->ie & IER_THR_EMPTY) {
        s->irq_id = DUO_IRQ_THR_EMPTY;
        cond = true;
    }

    if (s->rx_fifo_len > 0) {
        s->lsr |= LSR_RX_DATA_READY;
        if (s->fcr & FCR_FIFO_ENABLE) {
            s->usr |= USR_UART_RX_NOT_EMPTY | USR_UART_BUSY;
            switch (FCR_FIFO_RX_FULL(s->fcr)) {
            case RX_ONE_CHAR:
                rx_full = true;
                break;
            case RX_QUAT_FIFO:
                rx_full = s->rx_fifo_len >= DUO_UART_RX_FIFO_SIZE / 4;
                break;
            case RX_HALF_FIFO:
                rx_full = s->rx_fifo_len >= DUO_UART_RX_FIFO_SIZE / 2;
                break;
            case RX_2LESS_FULL_FIFO:
                rx_full = s->rx_fifo_len >= DUO_UART_RX_FIFO_SIZE - 2;
                break;
            default:
                qemu_build_not_reached();
                break;
            }

            if (rx_full) {
                s->usr |= USR_UART_RX_FULL;
            }
        }
        if (s->ie & IER_RX_READY) {
            s->irq_id = DUO_IRQ_RX_AVAIL;
            cond = true;
        }
    }

    if (cond) {
        qemu_irq_raise(s->irq);
    } else {
        s->irq_id = 1;
        qemu_irq_lower(s->irq);
    }
}

static uint64_t duo_uart_read(void *opaque, hwaddr addr, unsigned int size)
{
    DuoUARTState *s = opaque;
    unsigned char r;
    switch (addr) {
    case DUO_UART_RBR_THR_DLL:
        if (s->lcr & LCR_DIVISOR_ACCESS_BIT) {
            r = s->div & 0xFF;
            return r;
        } else if (s->rx_fifo_len) {
            r = s->rx_fifo[0];
            memmove(s->rx_fifo, s->rx_fifo + 1, s->rx_fifo_len - 1);
            s->rx_fifo_len--;
            qemu_chr_fe_accept_input(&s->chr);
            duo_uart_update_status(s);
            return r;
        }
        return 0x80000000;

    case DUO_UART_IER_DLH:
        if (s->lcr & LCR_DIVISOR_ACCESS_BIT) {
            r = s->div >> 8;
            return r;
        } else {
            return s->ie;
        }
    case DUO_UART_FCR_IIR:
        return (s->irq_id & 0xF) | (s->fcr & FCR_FIFO_ENABLE ? 0xC0 : 0);
    case DUO_UART_LCR:
        return s->lcr;
    case DUO_UART_MCR:
        return s->mcr;
    case DUO_UART_LSR:
        return s->lsr;
    case DUO_UART_MSR:
        return 0;
    case DUO_UART_LPDLL:
        if (s->lcr & LCR_DIVISOR_ACCESS_BIT) {
            r = s->lpdiv & 0xFF;
            return r;
        }
        break;
    case DUO_UART_LPDLH:
        if (s->lcr & LCR_DIVISOR_ACCESS_BIT) {
            r = (s->lpdiv >> 8) & 0xFF;
            return r;
        }
        break;
    case DUO_UART_SRBR_STHR:
        if (!(s->lcr & LCR_DIVISOR_ACCESS_BIT)) {
            if (s->rx_fifo_len) {
                r = s->rx_fifo[0];
                memmove(s->rx_fifo, s->rx_fifo + 1, s->rx_fifo_len - 1);
                s->rx_fifo_len--;
                qemu_chr_fe_accept_input(&s->chr);
                duo_uart_update_status(s);
                return r;
            }
            return 0x80000000;
        }
        break;
    case DUO_UART_USR:
        return s->usr;
    case DUO_UART_TFL:
        return 0;
    case DUO_UART_RFL:
        return s->rx_fifo_len;
    case DUO_UART_SRTS:
        return (s->mcr & 2) >> 1;
    case DUO_UART_SBCR:
        return (s->lcr & 0x40) >> 6;
    case DUO_UART_SDMAM:
        return (s->fcr & 0x8) >> 3;
    case DUO_UART_SFE:
        return s->fcr & FCR_FIFO_ENABLE;
    case DUO_UART_SRT:
        return (s->fcr & 0xC0) >> 6;
    case DUO_UART_STET:
        return (s->fcr & 0x30) >> 4;
    case DUO_UART_FAR:
    case DUO_UART_HTX:
    case DUO_UART_DMASA:
    case DUO_UART_RFW:
    case DUO_UART_SRR:
        return 0;
    }
    qemu_log_mask(LOG_GUEST_ERROR, "%s: bad read: addr=0x%x\n",
                  __func__, (int)addr);
    return 0;
}

static void duo_uart_write(void *opaque, hwaddr addr, uint64_t val64,
                           unsigned int size)
{
    DuoUARTState *s = opaque;
    uint32_t value = val64;
    unsigned char ch = value;

    switch (addr) {
    case DUO_UART_RBR_THR_DLL:
        if (s->lcr & LCR_DIVISOR_ACCESS_BIT) {
            s->div = (s->div & 0xFF00) | ch;
        } else {
            qemu_chr_fe_write(&s->chr, &ch, 1);
            duo_uart_update_status(s);
        }
        return;
    case DUO_UART_IER_DLH:
        if (s->lcr & LCR_DIVISOR_ACCESS_BIT) {
            s->div = (s->div & 0xFF) | (ch << 8);
        } else {
            s->ie = val64;
            duo_uart_update_status(s);
        }
        return;
    case DUO_UART_FCR_IIR:
        s->fcr = ch;
        if (s->fcr & 0x2) {  /* reset RX FIFO */
            s->rx_fifo_len = 0;
        }
        duo_uart_update_status(s);
        return;
    case DUO_UART_LCR:
        s->lcr = ch;
        return;
    case DUO_UART_MCR:
        s->mcr = ch;
        return;
    case DUO_UART_LPDLL:
        if (s->lcr & LCR_DIVISOR_ACCESS_BIT) {
            s->div = (s->lpdiv & 0xFF00) | ch;
            return;
        }
        break;
    case DUO_UART_LPDLH:
        if (s->lcr & LCR_DIVISOR_ACCESS_BIT) {
            s->lpdiv = (s->div & 0xFF) | (ch << 8);
            return;
        }
        break;
    case DUO_UART_SRBR_STHR:
        if (!(s->lcr & LCR_DIVISOR_ACCESS_BIT)) {
            qemu_chr_fe_write(&s->chr, &ch, 1);
            duo_uart_update_status(s);
            return;
        }
        break;
    case DUO_UART_SRTS:
        s->mcr = (s->mcr & ~2) | ((ch & 1) << 1);
        return;
    case DUO_UART_SBCR:
        s->lcr = (s->lcr & ~0x40) | ((ch & 1) << 6);
        return;
    case DUO_UART_SDMAM:
        s->fcr = (s->fcr & ~8) | ((ch & 1) << 3);
        duo_uart_update_status(s);
        return;
    case DUO_UART_SFE:
        s->fcr = (s->fcr & ~1) | (ch & 1);
        duo_uart_update_status(s);
        return;
    case DUO_UART_SRT:
        s->fcr = (s->fcr & ~0xC0) | ((ch & 3) << 6);
        duo_uart_update_status(s);
        return;
    case DUO_UART_STET:
        s->fcr = (s->fcr & ~0x30) | ((ch & 3) << 4);
        duo_uart_update_status(s);
        return;
    case DUO_UART_FAR:
    case DUO_UART_HTX:
    case DUO_UART_TFR:
    case DUO_UART_RFW:
    case DUO_UART_SRR:
    case DUO_UART_DMASA:
        break;
    }
    qemu_log_mask(LOG_GUEST_ERROR, "%s: bad write: addr=0x%x v=0x%x\n",
                  __func__, (int)addr, (int)value);
}

static const MemoryRegionOps duo_uart_ops = {
    .read = duo_uart_read,
    .write = duo_uart_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4
    }
};

static void duo_uart_rx(void *opaque, const uint8_t *buf, int size)
{
    DuoUARTState *s = opaque;

    /* Got a byte.  */
    if (s->rx_fifo_len >= sizeof(s->rx_fifo)) {
        printf("WARNING: UART dropped char.\n");
        return;
    }
    s->rx_fifo[s->rx_fifo_len++] = *buf;

    duo_uart_update_status(s);
}

static int duo_uart_can_rx(void *opaque)
{
    DuoUARTState *s = opaque;

    return s->rx_fifo_len < sizeof(s->rx_fifo);
}

static void duo_uart_event(void *opaque, QEMUChrEvent event)
{
}

static int duo_uart_be_change(void *opaque)
{
    DuoUARTState *s = opaque;

    qemu_chr_fe_set_handlers(&s->chr, duo_uart_can_rx, duo_uart_rx,
                             duo_uart_event, duo_uart_be_change, s,
                             NULL, true);

    return 0;
}

static Property duo_uart_properties[] = {
    DEFINE_PROP_CHR("chardev", DuoUARTState, chr),
    DEFINE_PROP_END_OF_LIST(),
};

static void duo_uart_init(Object *obj)
{
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);
    DuoUARTState *s = DUO_UART(obj);

    memory_region_init_io(&s->mmio, OBJECT(s), &duo_uart_ops, s,
                          TYPE_DUO_UART, DUO_UART_MAX);
    sysbus_init_mmio(sbd, &s->mmio);
    sysbus_init_irq(sbd, &s->irq);
}

static void duo_uart_realize(DeviceState *dev, Error **errp)
{
    DuoUARTState *s = DUO_UART(dev);

    qemu_chr_fe_set_handlers(&s->chr, duo_uart_can_rx, duo_uart_rx,
                             duo_uart_event, duo_uart_be_change, s,
                             NULL, true);

}

static void duo_uart_reset_enter(Object *obj, ResetType type)
{
    DuoUARTState *s = DUO_UART(obj);
    s->ie = 0;
    s->div = 0;
    s->lcr = 0;
    s->irq_id = 1;
    s->fcr = 0;
    s->mcr = 0;
    s->lsr = 0;
    s->lpdiv = 0;
    s->usr = 0;
    s->rx_fifo_len = 0;
}

static void duo_uart_reset_hold(Object *obj)
{
    DuoUARTState *s = DUO_UART(obj);
    qemu_irq_lower(s->irq);
}

static const VMStateDescription vmstate_duo_uart = {
    .name = TYPE_DUO_UART,
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (VMStateField[]) {
        VMSTATE_UINT8_ARRAY(rx_fifo, DuoUARTState,
                            DUO_UART_RX_FIFO_SIZE),
        VMSTATE_UINT8(rx_fifo_len, DuoUARTState),
        VMSTATE_UINT8(ie, DuoUARTState),
        VMSTATE_UINT16(div, DuoUARTState),
        VMSTATE_UINT8(lcr, DuoUARTState),
        VMSTATE_UINT8(irq_id, DuoUARTState),
        VMSTATE_UINT8(fcr, DuoUARTState),
        VMSTATE_UINT8(mcr, DuoUARTState),
        VMSTATE_UINT8(lsr, DuoUARTState),
        VMSTATE_UINT16(lpdiv, DuoUARTState),
        VMSTATE_UINT8(usr, DuoUARTState),
        VMSTATE_END_OF_LIST()
    },
};


static void duo_uart_class_init(ObjectClass *oc, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(oc);
    ResettableClass *rc = RESETTABLE_CLASS(oc);

    dc->realize = duo_uart_realize;
    dc->vmsd = &vmstate_duo_uart;
    rc->phases.enter = duo_uart_reset_enter;
    rc->phases.hold  = duo_uart_reset_hold;
    device_class_set_props(dc, duo_uart_properties);
    set_bit(DEVICE_CATEGORY_INPUT, dc->categories);
}

static const TypeInfo duo_uart_info = {
    .name          = TYPE_DUO_UART,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(DuoUARTState),
    .instance_init = duo_uart_init,
    .class_init    = duo_uart_class_init,
};

static void duo_uart_register_types(void)
{
    type_register_static(&duo_uart_info);
}

type_init(duo_uart_register_types)

/*
 * Create UART device.
 */
DuoUARTState *duo_uart_create(MemoryRegion *address_space, hwaddr base,
    Chardev *chr, qemu_irq irq)
{
    DeviceState *dev;
    SysBusDevice *s;

    dev = qdev_new("riscv.duo.uart");
    s = SYS_BUS_DEVICE(dev);
    qdev_prop_set_chr(dev, "chardev", chr);
    sysbus_realize_and_unref(s, &error_fatal);
    memory_region_add_subregion(address_space, base,
                                sysbus_mmio_get_region(s, 0));
    sysbus_connect_irq(s, 0, irq);

    return DUO_UART(dev);
}
