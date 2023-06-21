/* SPDX-License-Identifier: MIT */
/*
 * Define RISC-V target-specific operand constraints.
 * Copyright (c) 2021 Linaro
 */

/*
 * Define constraint letters for register sets:
 * REGS(letter, register_mask)
 */
REGS('r', ALL_GENERAL_REGS)
REGS('v', ALL_VECTOR_REGS)
REGS('V', VECTOR_REGS_WO_V0)
/*
 * Defining constraint letters for vector register grouping:
 * aka. LMUL != 1
 */
REGS('d', ALL_DVECTOR_REG_GROUPS)
REGS('q', ALL_QVECTOR_REG_GROUPS)

/*
 * Define constraint letters for constants:
 * CONST(letter, TCG_CT_CONST_* bit set)
 */
CONST('I', TCG_CT_CONST_S12)
CONST('J', TCG_CT_CONST_J12)
CONST('N', TCG_CT_CONST_N12)
CONST('M', TCG_CT_CONST_M12)
CONST('Z', TCG_CT_CONST_ZERO)
CONST('U', TCG_CT_CONST_U4)
