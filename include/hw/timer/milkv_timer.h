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
#ifndef HW_TIMER_MILKV_TIMER_H
#define HW_TIMER_MILKV_TIMER_H

#include "hw/sysbus.h"
#include "qom/object.h"

#define DUO_TIMER_MAX_NUM 8

/* State of a single timer or watchdog block */
typedef struct {
    uint32_t load_count;
    uint32_t control;
    uint32_t status;
    struct ptimer_state *timer;
    qemu_irq irq;
    MemoryRegion iomem;
} DuoTimerBlock;

#define TYPE_DUO_TIMER "duo_timer"
OBJECT_DECLARE_SIMPLE_TYPE(DuoTimerState, DUO_TIMER)

struct DuoTimerState {
    /*< private >*/
    SysBusDevice parent_obj;
    /*< public >*/

    DuoTimerBlock timerblock[DUO_TIMER_MAX_NUM];
    MemoryRegion iomem;
};

#endif
