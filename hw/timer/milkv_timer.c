/*
 * Milkv DUO Timer
 *
 * Copyright (c) 2023  PLCT Lab
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version
 * 2 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#include "qemu/osdep.h"
#include "hw/hw.h"
#include "hw/irq.h"
#include "hw/ptimer.h"
#include "hw/qdev-properties.h"
#include "hw/timer/milkv_timer.h"
#include "migration/vmstate.h"
#include "qapi/error.h"
#include "qemu/module.h"
#include "hw/core/cpu.h"

#define PTIMER_POLICY                       \
    (PTIMER_POLICY_WRAP_AFTER_ONE_PERIOD |  \
     PTIMER_POLICY_CONTINUOUS_TRIGGER    |  \
     PTIMER_POLICY_NO_IMMEDIATE_TRIGGER  |  \
     PTIMER_POLICY_NO_IMMEDIATE_RELOAD   |  \
     PTIMER_POLICY_NO_COUNTER_ROUND_DOWN)


static inline void timerblock_update_irq(DuoTimerBlock *tb)
{
    qemu_set_irq(tb->irq, tb->status && !(tb->control & 4));
}

static void timerblock_tick(void *opaque)
{
    DuoTimerBlock *tb = (DuoTimerBlock *)opaque;

    if ((tb->control & 2) && ptimer_get_limit(tb->timer) == 0) {
        ptimer_stop(tb->timer);
    }
    tb->status = 1;
    timerblock_update_irq(tb);
}

static uint64_t timerblock_read(void *opaque, hwaddr addr,
                                unsigned size)
{
    DuoTimerBlock *tb = (DuoTimerBlock *)opaque;
    switch (addr) {
    case 0: /* LoadCount */
        return tb->load_count;
    case 4: /* CurrentValue */
        return ptimer_get_count(tb->timer);
    case 8: /* ControlReg */
        return tb->control;
    case 12: /* EOI */
        tb->status = 0;
        qemu_set_irq(tb->irq, 0);
        return tb->status;
    case 16: /* IntStatus */
        return tb->status && !(tb->control & 4);
    default:
        return 0;
    }
}

static void timerblock_write(void *opaque, hwaddr addr,
                             uint64_t value, unsigned size)
{
    DuoTimerBlock *tb = (DuoTimerBlock *)opaque;
    uint32_t control = tb->control;
    switch (addr) {
    case 0: /* LoadCount */
        ptimer_transaction_begin(tb->timer);
        tb->load_count = value;
        ptimer_set_count(tb->timer, value);
        ptimer_transaction_commit(tb->timer);
        break;
    case 8: /* ControlReg */
        ptimer_transaction_begin(tb->timer);
        if ((control & 3) != (value & 3)) {
            ptimer_stop(tb->timer);
            ptimer_set_limit(tb->timer, value & 2 ? tb->load_count :
                                                    UINT32_MAX, 0);
            if (value & 1) {
                ptimer_run(tb->timer, !(control & 2));
            }
        }
        tb->control = value;
        ptimer_transaction_commit(tb->timer);
        break;
    default:
        return;
    }
}

static uint64_t duo_timer_read(void *opaque, hwaddr addr, unsigned size)
{
    DuoTimerState *s = (DuoTimerState *)opaque;
    if (addr >= 0xa0) {
        uint8_t value = 0;
        uint8_t i;

        for (i = 0; i < DUO_TIMER_MAX_NUM; i++) {
            DuoTimerBlock *tb = &s->timerblock[i];
            value = value << 1;
            switch (addr) {
            case 0xa0: /* TimersIntStatus */
                value |= tb->status && !(tb->control & 4);
                break;
            case 0xa4: /* TimersEOI */
                qemu_set_irq(tb->irq, 0);
                value = 0;
                break;
            case 0xa8: /* TimersRawIntStatus */
                value |= tb->status && !(tb->control & 4);
                break;
            default:
                return 0;
            }
        }
        return value;
    } else {
        return timerblock_read(&s->timerblock[addr / 0x14], addr % 0x14, size);
    }
}

static void duo_timer_write(void *opaque, hwaddr addr, uint64_t value,
                            unsigned size)
{
    DuoTimerState *s = (DuoTimerState *)opaque;
    if (addr < 0xa0) {
        timerblock_write(&s->timerblock[addr / 0x14], addr % 0x14, value,
                         size);
    }
}

static const MemoryRegionOps duo_timer_ops = {
    .read = duo_timer_read,
    .write = duo_timer_write,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
    .endianness = DEVICE_NATIVE_ENDIAN,
};

static const MemoryRegionOps timerblock_ops = {
    .read = timerblock_read,
    .write = timerblock_write,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
    .endianness = DEVICE_NATIVE_ENDIAN,
};

static void timerblock_reset(DuoTimerBlock *tb)
{
    tb->control = 0;
    tb->status = 0;
    if (tb->timer) {
        ptimer_transaction_begin(tb->timer);
        ptimer_stop(tb->timer);
        ptimer_set_limit(tb->timer, 0, 1);
        ptimer_set_freq(tb->timer, 32000);
        ptimer_transaction_commit(tb->timer);
    }
}

static void duo_timer_reset(DeviceState *dev)
{
    DuoTimerState *s = DUO_TIMER(dev);
    int i;

    for (i = 0; i < ARRAY_SIZE(s->timerblock); i++) {
        timerblock_reset(&s->timerblock[i]);
    }
}

static void duo_timer_init(Object *obj)
{
    DuoTimerState *s = DUO_TIMER(obj);

    memory_region_init_io(&s->iomem, obj, &duo_timer_ops, s, "duo_timer",
                          0x14);
    sysbus_init_mmio(SYS_BUS_DEVICE(obj), &s->iomem);
}

static void duo_timer_realize(DeviceState *dev, Error **errp)
{
    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);
    DuoTimerState *s = DUO_TIMER(dev);
    int i;

    for (i = 0; i < DUO_TIMER_MAX_NUM; i++) {
        DuoTimerBlock *tb = &s->timerblock[i];
        tb->timer = ptimer_init(timerblock_tick, tb, PTIMER_POLICY);
        sysbus_init_irq(sbd, &tb->irq);
        memory_region_init_io(&tb->iomem, OBJECT(s), &timerblock_ops, tb,
                              "duo_timer_timerblock", 0x14);
        sysbus_init_mmio(sbd, &tb->iomem);
    }
}

static const VMStateDescription vmstate_timerblock = {
    .name = "duo_timer_timerblock",
    .version_id = 3,
    .minimum_version_id = 3,
    .fields = (VMStateField[]) {
        VMSTATE_UINT32(load_count, DuoTimerBlock),
        VMSTATE_UINT32(control, DuoTimerBlock),
        VMSTATE_UINT32(status, DuoTimerBlock),
        VMSTATE_PTIMER(timer, DuoTimerBlock),
        VMSTATE_END_OF_LIST()
    }
};

static const VMStateDescription vmstate_duo_timer = {
    .name = "duo_timer",
    .version_id = 3,
    .minimum_version_id = 3,
    .fields = (VMStateField[]) {
        VMSTATE_STRUCT_ARRAY(timerblock, DuoTimerState, DUO_TIMER_MAX_NUM, 3,
                             vmstate_timerblock, DuoTimerBlock),
        VMSTATE_END_OF_LIST()
    }
};


static void duo_timer_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->realize = duo_timer_realize;
    dc->vmsd = &vmstate_duo_timer;
    dc->reset = duo_timer_reset;
}

static const TypeInfo duo_timer_info = {
    .name          = TYPE_DUO_TIMER,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(DuoTimerState),
    .instance_init = duo_timer_init,
    .class_init    = duo_timer_class_init,
};

static void duo_timer_register_types(void)
{
    type_register_static(&duo_timer_info);
}

type_init(duo_timer_register_types)
