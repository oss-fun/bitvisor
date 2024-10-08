/*
 * Copyright (c) 2007, 2008 University of Tsukuba
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

#include <arch/mm.h>
#include <arch/vmm_mem.h>
#include <constants.h>
#include <core/assert.h>
#include <core/mm.h>
#include <core/panic.h>
#include <core/printf.h>
#include <core/spinlock.h>
#include <core/string.h>
#include "../mm.h"
#include "../phys.h"
#include "ap.h"
#include "asm.h"
#include "constants.h"
#include "mm.h"
#include "pcpu.h"
#include "pmap.h"
#include "vmm_mem.h"

#define MAPMEM_ADDR_START	0xD0000000
#define MAPMEM_ADDR_END		0xFF000000

#ifdef __x86_64__
#	define MIN_HPHYS_LEN		(8UL * 1024 * 1024 * 1024)
#	define PAGE1GB_HPHYS_LEN	(512UL * 1024 * 1024 * 1024)
#	define HPHYS_ADDR		(1ULL << (12 + 9 + 9 + 9))
#else
#	define MIN_HPHYS_LEN		(MAPMEM_ADDR_START - HPHYS_ADDR)
#	define HPHYS_ADDR		0x80000000
#endif

struct mm_arch_proc_desc {
	phys_t pd_addr;
};

static spinlock_t mm_lock_process_virt_to_phys;
static spinlock_t mapmem_lock;
static virt_t mapmem_lastvirt;
static phys_t process_virt_to_phys_pdp_phys;
static u64 *process_virt_to_phys_pdp;
static u64 hphys_len;
static u64 phys_blank;

static void process_create_initial_map (void *virt, phys_t phys);
static void process_virt_to_phys_prepare (void);

static u64 as_translate_hphys (void *data, unsigned int *npages, u64 address);
static struct mm_as _as_hphys = {
	.translate = as_translate_hphys,
};
const struct mm_as *const as_hphys = &_as_hphys;

static u64 as_translate_passvm (void *data, unsigned int *npages, u64 address);
static struct mm_as _as_passvm = {
	.translate = as_translate_passvm,
};
const struct mm_as *const as_passvm = &_as_passvm;

static bool
page1gb_available (void)
{
	u32 a, b, c, d;

	asm_cpuid (CPUID_EXT_0, 0, &a, &b, &c, &d);
	if (a < CPUID_EXT_1)
		return false;
	asm_cpuid (CPUID_EXT_1, 0, &a, &b, &c, &d);
	if (!(d & CPUID_EXT_1_EDX_PAGE1GB_BIT))
		return false;
	return true;
}

static void
map_hphys_sub (pmap_t *m, u64 hphys_addr, u64 attr, u64 attrmask, ulong size,
	       int level)
{
	u64 pde;
	ulong addr;

	addr = hphys_len;
	while (addr >= size) {
		addr -= size;
		pmap_seek (m, hphys_addr + addr, level);
		pmap_autoalloc (m);
		pde = pmap_read (m);
		if ((pde & PDE_P_BIT) && (pde & PDE_ADDR_MASK64) != addr)
			panic ("map_hphys: error");
		pde = addr | attr;
		pmap_write (m, pde, attrmask);
	}
	if (addr)
		panic ("map_hphys: hphys_len %llu is bad", hphys_len);
}

static void
map_hphys (void)
{
	ulong cr3, size;
	u64 attr, attrmask;
	pmap_t m;
	int level;
	unsigned int i;

	asm_rdcr3 (&cr3);
	pmap_open_vmm (&m, cr3, PMAP_LEVELS);
	hphys_len = MIN_HPHYS_LEN;
	attr = PDE_P_BIT | PDE_RW_BIT | PDE_PS_BIT | PDE_G_BIT;
	attrmask = attr | PDE_US_BIT | PDE_PWT_BIT | PDE_PCD_BIT |
		PDE_PS_PAT_BIT;
	size = PAGESIZE2M;
	level = 2;
	if (page1gb_available ()) {
#ifdef __x86_64__
		hphys_len = PAGE1GB_HPHYS_LEN;
		size = PAGESIZE1G;
		level = 3;
#endif
	}
	for (i = 0; i < 8; i++) {
		map_hphys_sub (&m, HPHYS_ADDR + hphys_len * i,
			       ((i & 1) ? PDE_PWT_BIT : 0) |
			       ((i & 2) ? PDE_PCD_BIT : 0) |
			       ((i & 4) ? PDE_PS_PAT_BIT : 0) | attr,
			       attrmask, size, level);
#ifndef __x86_64__
		/* 32bit address space is not enough for mapping a lot
		 * of uncached pages */
		break;
#endif
	}
	pmap_close (&m);
}

void
mm_arch_init (void)
{
	void *tmp;

	spinlock_init (&mm_lock_process_virt_to_phys);
	spinlock_init (&mapmem_lock);
	alloc_page (&tmp, &phys_blank);
	memset (tmp, 0, PAGESIZE);
	mapmem_lastvirt = MAPMEM_ADDR_START;
	map_hphys ();
	vmm_mem_unmap_user_area ();	/* for detecting null pointer */
	process_virt_to_phys_prepare ();
}

void
mm_arch_force_unlock (void)
{
	spinlock_unlock (&mm_lock_process_virt_to_phys);
	spinlock_unlock (&mapmem_lock);
}

/*** process ***/

static void
process_create_initial_map (void *virt, phys_t phys)
{
	u32 *pde;
	int i;

	pde = virt;
	/* clear PDEs for user area */
	for (i = 0; i < 0x400; i++)
		pde[i] = 0;
}

int
mm_process_arch_alloc (struct mm_arch_proc_desc **mm_proc_desc_out,
		       int space_id)
{
	void *virt;
	phys_t phys;
	struct mm_arch_proc_desc *out;

	alloc_page (&virt, &phys);
	process_create_initial_map (virt, phys);
	out = alloc (sizeof *out);
	out->pd_addr = phys;
	*mm_proc_desc_out = out;
	return 0;
}

void
mm_process_arch_free (struct mm_arch_proc_desc *mm_proc_desc)
{
	ASSERT (mm_proc_desc);
	free_page_phys (mm_proc_desc->pd_addr);
	free (mm_proc_desc);
}

void
mm_process_arch_mappage (struct mm_arch_proc_desc *mm_proc_desc, virt_t virt,
			 phys_t phys, u64 flags)
{
	ulong cr3;
	pmap_t m;
	u64 pte;

	pte = PTE_P_BIT | PTE_US_BIT;
	if (flags & MM_PROCESS_MAP_WRITE)
		pte |= PTE_RW_BIT;
	if (flags & MM_PROCESS_MAP_SHARE)
		pte |= PTE_AVAILABLE2_BIT;
	pte = phys | pte;
	asm_rdcr3 (&cr3);
	pmap_open_vmm (&m, cr3, PMAP_LEVELS);
	pmap_seek (&m, virt, 1);
	pmap_autoalloc (&m);
	ASSERT (!(pmap_read (&m) & PTE_P_BIT));
	pmap_write (&m, pte, 0xFFF);
	pmap_close (&m);
	asm_wrcr3 (cr3);
}

int
mm_process_arch_mapstack (struct mm_arch_proc_desc *mm_proc_desc, virt_t virt,
			  bool noalloc)
{
	ulong cr3;
	pmap_t m;
	u64 pte;
	void *tmp;
	phys_t phys;

	asm_rdcr3 (&cr3);
	pmap_open_vmm (&m, cr3, PMAP_LEVELS);
	pmap_seek (&m, virt, 1);
	pmap_autoalloc (&m);
	pte = pmap_read (&m);
	if (!(pte & PTE_P_BIT)) {
		if (noalloc)
			return -1;
		alloc_page (&tmp, &phys);
		memset (tmp, 0, PAGESIZE);
		pte = phys | PTE_P_BIT | PTE_RW_BIT | PTE_US_BIT;
	} else {
		pte &= ~PTE_AVAILABLE1_BIT;
		if (pte & PTE_A_BIT) {
			printf ("warning: user stack 0x%lX is accessed.\n",
				virt);
			pte &= ~PTE_A_BIT;
		}
		if (pte & PTE_D_BIT) {
			printf ("warning: user stack 0x%lX is dirty.\n", virt);
			pte &= ~PTE_D_BIT;
		}
	}
	pmap_write (&m, pte, 0xFFF);
	pmap_close (&m);
	asm_wrcr3 (cr3);
	return 0;
}

bool
mm_process_arch_shared_mem_absent (struct mm_arch_proc_desc *mm_proc_desc,
				   virt_t virt)
{
	pmap_t m;
	ulong cr3;
	u64 pte;

	asm_rdcr3 (&cr3);
	pmap_open_vmm (&m, cr3, PMAP_LEVELS);
	pmap_seek (&m, virt, 1);
	pte = pmap_read (&m);
	pmap_close (&m);
	if (!(pte & PTE_P_BIT))
		return true;
	return false;
}

static void
process_virt_to_phys_prepare (void)
{
	void *virt;

	alloc_page (&virt, &process_virt_to_phys_pdp_phys);
	memset (virt, 0, 0x20);
	process_virt_to_phys_pdp = virt;
}

int
mm_process_arch_virt_to_phys (struct mm_arch_proc_desc *mm_proc_desc,
			      virt_t virt, phys_t *phys, bool expect_writable)
{
	u64 pte;
	int r = -1;
	pmap_t m;

	spinlock_lock (&mm_lock_process_virt_to_phys);
	if (virt < 0x40000000) {
		*process_virt_to_phys_pdp = mm_proc_desc->pd_addr | PDE_P_BIT;
		pmap_open_vmm (&m, process_virt_to_phys_pdp_phys, 3);
	} else {
		ulong cr3;

		asm_rdcr3 (&cr3);
		pmap_open_vmm (&m, cr3, PMAP_LEVELS);
	}
	pmap_seek (&m, virt, 1);
	pte = pmap_read (&m);
	if (expect_writable) {
		if (!(pte & PTE_RW_BIT))
			goto end;
	}
	if (pte & PTE_P_BIT) {
		*phys = (pte & PTE_ADDR_MASK) | (virt & ~PTE_ADDR_MASK);
		r = 0;
	}
end:
	pmap_close (&m);
	spinlock_unlock (&mm_lock_process_virt_to_phys);
	return r;
}

bool
mm_process_arch_stack_absent (struct mm_arch_proc_desc *mm_proc_desc,
			      virt_t virt)
{
	pmap_t m;
	ulong cr3;
	u64 pte;

	asm_rdcr3 (&cr3);
	pmap_open_vmm (&m, cr3, PMAP_LEVELS);
	pmap_seek (&m, virt, 1);
	pte = pmap_read (&m);
	pmap_close (&m);
	if (!(pte & PTE_P_BIT))
		return true;
	if (pte & PTE_AVAILABLE1_BIT)
		return true;
	return false;
}

int
mm_process_arch_unmap (struct mm_arch_proc_desc *mm_proc_desc,
		       virt_t aligned_virt, uint npages)
{
	virt_t v;
	u64 pte;
	ulong cr3;
	pmap_t m;

	asm_rdcr3 (&cr3);
	pmap_open_vmm (&m, cr3, PMAP_LEVELS);
	for (v = aligned_virt; npages > 0; v += PAGESIZE, npages--) {
		pmap_seek (&m, v, 1);
		pte = pmap_read (&m);
		if (!(pte & PTE_P_BIT))
			continue;
		if (!(pte & PTE_AVAILABLE2_BIT)) /* if not shared memory */
			free_page_phys (pte);
		pmap_write (&m, 0, 0);
	}
	pmap_close (&m);
	asm_wrcr3 (cr3);
	return 0;
}


int
mm_process_arch_unmap_stack (struct mm_arch_proc_desc *mm_proc_desc,
			     virt_t aligned_virt, uint npages)
{
	virt_t v;
	u64 pte;
	ulong cr3;
	pmap_t m;

	asm_rdcr3 (&cr3);
	pmap_open_vmm (&m, cr3, PMAP_LEVELS);
	for (v = aligned_virt; npages > 0; v += PAGESIZE, npages--) {
		pmap_seek (&m, v, 1);
		pte = pmap_read (&m);
		if (!(pte & PTE_P_BIT))
			continue;
		if (pte & PTE_AVAILABLE2_BIT) {
			printf ("shared page 0x%lX in stack?\n", v);
			continue;
		}
		pte &= ~(PTE_A_BIT | PTE_D_BIT);
		pmap_write (&m, pte | PTE_AVAILABLE1_BIT, 0xFFF);
	}
	/* checking about stack overflow/underflow */
	/* process is locked in process.c */
	pmap_seek (&m, v, 1);
	pte = pmap_read (&m);
	if ((pte & PTE_A_BIT) && /* accessed */
	    (pte & PTE_P_BIT) && /* present */
	    (pte & PTE_RW_BIT) && /* writable */
	    (pte & PTE_US_BIT) && /* user page */
	    (pte & PTE_AVAILABLE1_BIT) && /* unmapped stack page */
	    !(pte & PTE_AVAILABLE2_BIT)) { /* not shared page */
		printf ("warning: stack underflow detected."
			" page 0x%lX is %s.\n", v,
			(pte & PTE_D_BIT) ? "dirty" : "accessed");
		pte &= ~(PTE_A_BIT | PTE_D_BIT);
		pmap_write (&m, pte, 0xFFF);
	}
	pmap_seek (&m, aligned_virt - PAGESIZE, 1);
	pte = pmap_read (&m);
	if ((pte & PTE_A_BIT) && /* accessed */
	    (pte & PTE_P_BIT) && /* present */
	    (pte & PTE_RW_BIT) && /* writable */
	    (pte & PTE_US_BIT) && /* user page */
	    (pte & PTE_AVAILABLE1_BIT) && /* unmapped stack page */
	    !(pte & PTE_AVAILABLE2_BIT)) { /* not shared page */
		printf ("warning: stack overflow detected."
			" page 0x%lX is %s.\n", v,
			(pte & PTE_D_BIT) ? "dirty" : "accessed");
		pte &= ~(PTE_A_BIT | PTE_D_BIT);
		pmap_write (&m, pte, 0xFFF);
	}
	pmap_close (&m);
	asm_wrcr3 (cr3);
	return 0;
}

void
mm_process_arch_unmapall (struct mm_arch_proc_desc *mm_proc_desc)
{
	virt_t v;
	u64 pde;
	pmap_t m;
	ulong cr3;

	asm_rdcr3 (&cr3);
	pmap_open_vmm (&m, cr3, PMAP_LEVELS);
	for (v = 0; v < vmm_mem_proc_end_virt (); v += PAGESIZE2M) {
		pmap_seek (&m, v, 2);
		pde = pmap_read (&m);
		if (!(pde & PDE_P_BIT))
			continue;
		free_page_phys (pde);
		pmap_write (&m, 0, 0);
	}
	pmap_close (&m);
	asm_wrcr3 (cr3);
}

struct mm_arch_proc_desc *
mm_process_arch_switch (struct mm_arch_proc_desc *switchto)
{
	u64 old, new;
	struct mm_arch_proc_desc *ret;
	phys_t old_phys, new_phys;
	ulong cr3;
	pmap_t m;

	asm_rdcr3 (&cr3);
	pmap_open_vmm (&m, cr3, PMAP_LEVELS);
	pmap_seek (&m, 0, 3);
	old = pmap_read (&m);
	/* 1 is a special value for P=0 */
	if (!(old & PDE_P_BIT))
		old_phys = 1;
	else
		old_phys = old & ~PDE_ATTR_MASK;
	new_phys = switchto ? switchto->pd_addr : 1;
	if (new_phys != old_phys) {
		if (new_phys == 1)
			new = 0;
		else
			new = new_phys | PDE_P_BIT;
		pmap_write (&m, new, PDE_P_BIT);
		asm_wrcr3 (cr3);
	}
	pmap_close (&m);
	ret = currentcpu->cur_mm_proc_desc;
	currentcpu->cur_mm_proc_desc = switchto;
	return ret;
}

/**********************************************************************/
/*** Address space ***/

static u64
as_translate_hphys (void *data, unsigned int *npages, u64 address)
{
	return address | PTE_P_BIT | PTE_RW_BIT | PTE_US_BIT;
}

static u64
as_translate_passvm (void *data, unsigned int *npages, u64 address)
{
	u64 ret;
	unsigned int max_npages;

	/* Return dummy page if "phys" in VMM region. Dummy page is
	 * read-only, zero-filled page. See "mm_init_global()". */
	if (phys_in_vmm (address)) {
		ret = phys_blank | PTE_P_BIT | PTE_US_BIT;
		*npages = 1;
		return ret;
	}

	/* (address, *npages * PAGESIZE) region may be overlapped
	 * when 1GiB mapping. */
	if (phys_overlapping_with_vmm (address, *npages * PAGESIZE))
		*npages = (vmm_mem_start_phys () - address) / PAGESIZE;

	ret = mm_as_translate (as_hphys, npages, address);
	max_npages = (PAGESIZE1G - (address & PAGESIZE1G_MASK)) / PAGESIZE;
	if (*npages > max_npages)
		*npages = max_npages;
	return ret;
}

u64
mm_as_translate (const struct mm_as *as, unsigned int *npages, u64 address)
{
	unsigned int npage1 = 1;

	if (!npages)
		npages = &npage1;
	address &= ~PAGESIZE_MASK;
	return as->translate (as->data, npages, address);
}

u64
mm_as_msi_to_icr (const struct mm_as *as, u32 maddr, u32 mupper, u16 mdata)
{
	if (as->msi_to_icr)
		return as->msi_to_icr (as->data, maddr, mupper, mdata);
	return msi_to_icr (maddr, mupper, mdata);
}

/**********************************************************************/
/*** accessing memory ***/

static void *
mapped_hphys_addr (u64 hphys, uint len, int flags)
{
	u64 hphys_addr = HPHYS_ADDR;
	u64 hphys_len_copy = hphys_len;

	/* Expand MAPMEM_UC to MAPMEM_PCD | MAPMEM_PWT */
	if (flags & MAPMEM_UC)
		flags |= MAPMEM_PCD | MAPMEM_PWT;
#ifdef __x86_64__
	if (flags & MAPMEM_PAT)
		hphys_addr += hphys_len_copy * 4;
	if (flags & MAPMEM_PCD)
		hphys_addr += hphys_len_copy * 2;
	if (flags & MAPMEM_PWT)
		hphys_addr += hphys_len_copy;
#else
	if (flags & (MAPMEM_PWT | MAPMEM_PCD | MAPMEM_PAT))
		return NULL;
#endif
	if (hphys >= hphys_len_copy)
		return NULL;
	if (hphys + len - 1 >= hphys_len_copy)
		return NULL;
	return (void *)(virt_t)(hphys_addr + hphys);
}

static void *
mapped_as_addr (const struct mm_as *as, u64 phys, uint len, int flags)
{
	const u64 ptemask = ~PAGESIZE_MASK | PTE_P_BIT | PTE_RW_BIT;
	u64 hphys, pte, pteor, ptenext;
	unsigned int npages, n;

	npages = ((phys & PAGESIZE_MASK) + len + PAGESIZE - 1) >>
		PAGESIZE_SHIFT;
	n = npages;
	pteor = flags & MAPMEM_WRITE ? 0 : PTE_RW_BIT;
	pte = (mm_as_translate (as, &n, phys) | pteor) & ptemask;
	if (!(pte & PTE_P_BIT) || !(pte & PTE_RW_BIT))
		return NULL;
	hphys = (pte & ~PAGESIZE_MASK) | (phys & PAGESIZE_MASK);
	while (npages > n) {
		phys += n << PAGESIZE_SHIFT;
		ptenext = pte + (n << PAGESIZE_SHIFT);
		npages -= n;
		n = npages;
		pte = (mm_as_translate (as, &n, phys) | pteor) & ptemask;
		if (pte != ptenext)
			return NULL;
	}
	return mapped_hphys_addr (hphys, len, flags);
}

static void *
mapmem_alloc (pmap_t *m, uint offset, uint len)
{
	u64 pte;
	virt_t v;
	uint n, i;
	int loopcount = 0;

	n = (offset + len + PAGESIZE_MASK) >> PAGESIZE_SHIFT;
	spinlock_lock (&mapmem_lock);
	v = mapmem_lastvirt;
retry:
	for (i = 0; i < n; i++) {
		if (v + (i << PAGESIZE_SHIFT) >= MAPMEM_ADDR_END) {
			v = MAPMEM_ADDR_START;
			loopcount++;
			if (loopcount > 1) {
				spinlock_unlock (&mapmem_lock);
				panic ("%s: loopcount %d offset 0x%X len 0x%X",
				       __func__, loopcount, offset, len);
			}
			goto retry;
		}
		pmap_seek (m, v + (i << PAGESIZE_SHIFT), 1);
		pte = pmap_read (m);
		if (pte & PTE_P_BIT) {
			v = v + (i << PAGESIZE_SHIFT) + PAGESIZE;
			goto retry;
		}
	}
	for (i = 0; i < n; i++) {
		pmap_seek (m, v + (i << PAGESIZE_SHIFT), 1);
		pmap_autoalloc (m);
		pmap_write (m, PTE_P_BIT, 0xFFF);
	}
	mapmem_lastvirt = v + (n << PAGESIZE_SHIFT);
	spinlock_unlock (&mapmem_lock);
	return (void *)(v + offset);
}

static bool
mapmem_domap (pmap_t *m, void *virt, const struct mm_as *as, int flags,
	      u64 physaddr, uint len)
{
	virt_t v;
	u64 p, pte;
	uint n, i, offset;

	offset = physaddr & PAGESIZE_MASK;
	n = (offset + len + PAGESIZE_MASK) >> PAGESIZE_SHIFT;
	v = (virt_t)virt & ~PAGESIZE_MASK;
	p = physaddr & ~PAGESIZE_MASK;
	for (i = 0; i < n; i++) {
		pmap_seek (m, v + (i << PAGESIZE_SHIFT), 1);
		pte = mm_as_translate (as, NULL, p + (i << PAGESIZE_SHIFT));
		if (!(pte & PTE_P_BIT)) {
			if (!(flags & MAPMEM_CANFAIL))
				panic ("mapmem no page  as=%p%s"
				       " translate=%p data=%p address=0x%llX"
				       " pte=0x%llX", as,
				       as == as_hphys ? "(hphys)" :
				       as == as_passvm ? "(passvm)" : "",
				       as->translate, as->data,
				       p + (i << PAGESIZE_SHIFT), pte);
			return true;
		}
		if (!(pte & PTE_RW_BIT) && (flags & MAPMEM_WRITE)) {
			if (!(flags & MAPMEM_CANFAIL))
				panic ("mapmem readonly  as=%p%s"
				       " translate=%p data=%p address=0x%llX"
				       " pte=0x%llX", as,
				       as == as_hphys ? "(hphys)" :
				       as == as_passvm ? "(passvm)" : "",
				       as->translate, as->data,
				       p + (i << PAGESIZE_SHIFT), pte);
			return true;
		}
		pte = (pte & ~PAGESIZE_MASK) | PTE_P_BIT;
		if (flags & MAPMEM_UC)
			pte |= PTE_PCD_BIT | PTE_PWT_BIT;
		if (flags & MAPMEM_WRITE)
			pte |= PTE_RW_BIT;
		if (flags & MAPMEM_PWT)
			pte |= PTE_PWT_BIT;
		if (flags & MAPMEM_PCD)
			pte |= PTE_PCD_BIT;
		if (flags & MAPMEM_PAT)
			pte |= PTE_PAT_BIT;
		ASSERT (pmap_read (m) & PTE_P_BIT);
		pmap_write (m, pte, PTE_P_BIT | PTE_RW_BIT | PTE_US_BIT |
			    PTE_PWT_BIT | PTE_PCD_BIT | PTE_PAT_BIT);
		asm_invlpg ((void *)(v + (i << PAGESIZE_SHIFT)));
	}
	return false;
}

static void *
mapmem_internal (const struct mm_as *as, int flags, u64 physaddr, uint len)
{
	void *r;
	pmap_t m;
	ulong hostcr3;

	r = mapped_as_addr (as, physaddr, len, flags);
	if (r)
		return r;
	asm_rdcr3 (&hostcr3);
	pmap_open_vmm (&m, hostcr3, PMAP_LEVELS);
	r = mapmem_alloc (&m, physaddr & PAGESIZE_MASK, len);
	if (!r) {
		if (!(flags & MAPMEM_CANFAIL))
			panic ("mapmem_alloc failed  as=%p%s"
			       " translate=%p data=%p physaddr=0x%llX len=%u",
			       as, as == as_hphys ? "(hphys)" :
			       as == as_passvm ? "(passvm)" : "",
			       as->translate, as->data, physaddr, len);
		goto ret;
	}
	if (!mapmem_domap (&m, r, as, flags, physaddr, len))
		goto ret;
	unmapmem (r, len);
	r = NULL;
ret:
	ASSERT (r || (flags & MAPMEM_CANFAIL));
	pmap_close (&m);
	return r;
}

void *
mm_arch_mapmem_hphys (u64 physaddr, uint len, int flags)
{
	return mapmem_internal (as_hphys, flags, physaddr, len);
}

void *
mm_arch_mapmem_as (const struct mm_as *as, u64 physaddr, uint len, int flags)
{
	return mapmem_internal (as, flags, physaddr, len);
}

void
mm_arch_unmapmem (void *virt, uint len)
{
	pmap_t m;
	virt_t v;
	ulong hostcr3;
	uint n, i, offset;

	if ((virt_t)virt < MAPMEM_ADDR_START ||
	    (virt_t)virt >= MAPMEM_ADDR_END)
		return;
	spinlock_lock (&mapmem_lock);
	asm_rdcr3 (&hostcr3);
	pmap_open_vmm (&m, hostcr3, PMAP_LEVELS);
	offset = (virt_t)virt & PAGESIZE_MASK;
	n = (offset + len + PAGESIZE_MASK) >> PAGESIZE_SHIFT;
	v = (virt_t)virt & ~PAGESIZE_MASK;
	for (i = 0; i < n; i++) {
		pmap_seek (&m, v + (i << PAGESIZE_SHIFT), 1);
		if (pmap_read (&m) & PTE_P_BIT)
			pmap_write (&m, 0, 0);
		asm_invlpg ((void *)(v + (i << PAGESIZE_SHIFT)));
	}
	pmap_close (&m);
	spinlock_unlock (&mapmem_lock);
}

/* Flush all write back caches including other processors */
void
mm_flush_wb_cache (void)
{
	int tmp;

	/* Read all VMM memory to let other processors write back */
	asm volatile ("cld ; rep lodsl"
		      : "=a" (tmp), "=c" (tmp), "=S" (tmp)
		      : "S" (vmm_mem_start_virt ()), "c" (VMMSIZE_ALL / 4));
	asm_wbinvd ();		/* write back all caches */
}
