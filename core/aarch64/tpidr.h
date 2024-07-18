/*
 * Copyright (c) 2007, 2008 University of Tsukuba
 * Copyright (c) 2022 Igel Co., Ltd
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Tsukuba nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _CORE_AARCH64_TPIDR_H
#define _CORE_AARCH64_TPIDR_H

#include <core/types.h>
#include "asm.h"
#include "tpidr.h"

#define TPIDR_PANIC_STATE_MASK 0xFFULL

struct pcpu;

static inline u8
tpidr_get_panic_state (void)
{
	return mrs (TPIDR_EL2) & TPIDR_PANIC_STATE_MASK;
}

static inline void
tpidr_set_panic_state (u8 new_state)
{
	u64 pcpu = mrs (TPIDR_EL2) & ~TPIDR_PANIC_STATE_MASK;
	msr (TPIDR_EL2, pcpu | new_state);
}

static inline struct pcpu *
tpidr_get_pcpu (void)
{
	return (struct pcpu *)(mrs (TPIDR_EL2) & ~TPIDR_PANIC_STATE_MASK);
}

static inline void
tpidr_set_pcpu (struct pcpu *p)
{
	u8 panic_state = mrs (TPIDR_EL2) & TPIDR_PANIC_STATE_MASK;
	u64 pcpu = (u64)p & ~TPIDR_PANIC_STATE_MASK;
	msr (TPIDR_EL2, pcpu | panic_state);
}

#endif
