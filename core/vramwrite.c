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

#include <arch/vramwrite.h>
#include <core/process.h>
#include <core/string.h>
#include <core/types.h>
#include <core/vga.h>
#include "initfunc.h"
#include "mm.h"
#include "panic.h"
#include "uefi.h"
#include "vramwrite.h"

static struct vramwrite_font vf;
static u32 *scrollbuf;

#ifdef TTY_VGA
/* font data converted from mplus_f12r.bdf
M+ BITMAP FONTS            Copyright 2002-2005  COZ <coz@users.sourceforge.jp>
*/
static u8 mplus_f12r[] = {
	0x01, 0x7B, 0x7A, 0x79, 0x7B, 0x7A, 0x79, 0x7B, 0x7A, 0x00,
	0x00, 0x00, 0x00, 0x21, 0x73, 0xF8, 0x20, 0x00, 0x00, 0x00,
	0x01, 0x55, 0xA9, 0xA9, 0x55, 0xA9, 0xA9, 0x55, 0xA9, 0xA8,
	0x03, 0x8B, 0x8A, 0x8B, 0x8B, 0x02, 0x20, 0x22, 0x20, 0x00,
	0x03, 0xFB, 0x82, 0x83, 0x83, 0x02, 0x82, 0xF8, 0x80, 0x00,
	0x02, 0xF8, 0x88, 0x8B, 0xFB, 0x02, 0x8A, 0xF8, 0x92, 0x00,
	0x02, 0x80, 0x80, 0x83, 0xFB, 0x02, 0x82, 0xF8, 0x80, 0x00,
	0x02, 0x61, 0x90, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x02, 0x00, 0x20, 0xFA, 0x20, 0x00, 0x00, 0xF8, 0x00,
	0x02, 0x8A, 0xCA, 0x9A, 0x88, 0x00, 0x83, 0x83, 0x82, 0x00,
	0x02, 0x88, 0x8A, 0x93, 0xE3, 0x02, 0x20, 0x22, 0x20, 0x00,
	0x20, 0x22, 0x20, 0x23, 0x22, 0x20, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x03, 0x02, 0x00, 0x20, 0x22, 0x20, 0x20,
	0x00, 0x00, 0x00, 0x00, 0x03, 0x03, 0x20, 0x22, 0x20, 0x20,
	0x20, 0x22, 0x20, 0x20, 0x23, 0x23, 0x00, 0x00, 0x00, 0x00,
	0x20, 0x22, 0x20, 0x23, 0x23, 0x23, 0x20, 0x22, 0x20, 0x20,
	0x03, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0xFC, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x03, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFC, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x03, 0x03, 0x00,
	0x20, 0x22, 0x20, 0x20, 0x23, 0x23, 0x20, 0x22, 0x20, 0x20,
	0x20, 0x22, 0x20, 0x23, 0x22, 0x20, 0x20, 0x22, 0x20, 0x20,
	0x20, 0x22, 0x20, 0x23, 0x23, 0x23, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x03, 0x03, 0x03, 0x20, 0x22, 0x20, 0x20,
	0x20, 0x22, 0x20, 0x20, 0x22, 0x20, 0x20, 0x22, 0x20, 0x20,
	0x00, 0x02, 0x10, 0x40, 0x82, 0x40, 0x10, 0x00, 0xF8, 0x00,
	0x00, 0x02, 0x40, 0x10, 0x0A, 0x10, 0x40, 0x00, 0xF8, 0x00,
	0x00, 0x00, 0x00, 0x01, 0xF9, 0x50, 0x50, 0x90, 0x90, 0x00,
	0x00, 0x01, 0x00, 0x13, 0xFB, 0x22, 0x40, 0x40, 0x00, 0x00,
	0x01, 0x00, 0x32, 0x41, 0x40, 0xF0, 0x40, 0x60, 0x98, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x22, 0x70, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x02, 0x20, 0x20, 0x22, 0x20, 0x00, 0x00, 0x20, 0x00,
	0x01, 0x51, 0x50, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x01, 0x01, 0x50, 0xF9, 0x51, 0x50, 0xF8, 0x50, 0x50, 0x00,
	0x01, 0x03, 0x22, 0xA0, 0xA2, 0x72, 0x28, 0xF0, 0x20, 0x00,
	0x02, 0x02, 0x40, 0x49, 0x10, 0x20, 0x90, 0x28, 0x10, 0x00,
	0x02, 0x01, 0x60, 0x92, 0x62, 0x4A, 0x90, 0x90, 0x68, 0x00,
	0x00, 0x22, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x12, 0x20, 0x41, 0x40, 0x40, 0x40, 0x21, 0x20, 0x00,
	0x00, 0x42, 0x20, 0x10, 0x11, 0x10, 0x11, 0x20, 0x20, 0x00,
	0x00, 0x02, 0x00, 0xA9, 0x73, 0x20, 0xA8, 0x20, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x20, 0x22, 0xF8, 0x20, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x21, 0x70, 0x20, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x70, 0x20, 0x00,
	0x00, 0x00, 0x0A, 0x10, 0x12, 0x20, 0x42, 0x40, 0x80, 0x00,
	0x02, 0x00, 0x72, 0x8B, 0x98, 0xAA, 0x88, 0x88, 0x70, 0x00,
	0x01, 0x02, 0x20, 0xA0, 0x22, 0x20, 0x20, 0x20, 0x20, 0x00,
	0x02, 0x00, 0x72, 0x88, 0x0A, 0x10, 0x40, 0x80, 0xF8, 0x00,
	0x00, 0x00, 0xFA, 0x10, 0x20, 0x72, 0x08, 0x88, 0x70, 0x00,
	0x00, 0x03, 0x30, 0x52, 0x51, 0x90, 0xF8, 0x10, 0x10, 0x00,
	0x02, 0x00, 0xF8, 0x80, 0xF0, 0x0A, 0x88, 0x88, 0x70, 0x00,
	0x01, 0x00, 0x30, 0x82, 0xF0, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x00, 0xFA, 0x10, 0x12, 0x20, 0x40, 0x40, 0x40, 0x00,
	0x02, 0x00, 0x72, 0x89, 0x51, 0x20, 0x88, 0x88, 0x70, 0x00,
	0x02, 0x00, 0x72, 0x89, 0x8B, 0x8A, 0x08, 0x10, 0x60, 0x00,
	0x00, 0x00, 0x00, 0x20, 0x70, 0x20, 0x20, 0x70, 0x20, 0x00,
	0x00, 0x00, 0x00, 0x20, 0x70, 0x20, 0x21, 0x70, 0x20, 0x00,
	0x00, 0x01, 0x08, 0x21, 0x40, 0x80, 0x20, 0x10, 0x08, 0x00,
	0x00, 0x00, 0x00, 0x03, 0xFB, 0x02, 0x00, 0x00, 0x00, 0x00,
	0x01, 0x00, 0x80, 0x20, 0x11, 0x08, 0x20, 0x40, 0x80, 0x00,
	0x02, 0x00, 0x72, 0x88, 0x0A, 0x10, 0x20, 0x00, 0x20, 0x00,
	0x02, 0x00, 0x72, 0xBA, 0xAA, 0xAA, 0xB8, 0x80, 0x70, 0x00,
	0x00, 0x02, 0x20, 0x53, 0x53, 0x52, 0x88, 0x88, 0x88, 0x00,
	0x02, 0x00, 0xF2, 0x8A, 0x88, 0xF2, 0x88, 0x88, 0xF0, 0x00,
	0x02, 0x00, 0x72, 0x82, 0x80, 0x80, 0x80, 0x88, 0x70, 0x00,
	0x02, 0x01, 0xE0, 0x8A, 0x88, 0x8A, 0x88, 0x90, 0xE0, 0x00,
	0x02, 0x00, 0xF8, 0x82, 0x80, 0xF0, 0x80, 0x80, 0xF8, 0x00,
	0x02, 0x00, 0xF8, 0x82, 0x80, 0xF0, 0x80, 0x80, 0x80, 0x00,
	0x02, 0x00, 0x72, 0x82, 0x80, 0xBA, 0x88, 0x88, 0x78, 0x00,
	0x02, 0x00, 0x8A, 0x8A, 0x88, 0xFA, 0x88, 0x88, 0x88, 0x00,
	0x00, 0x02, 0x70, 0x20, 0x22, 0x20, 0x20, 0x20, 0x70, 0x00,
	0x00, 0x00, 0x0A, 0x0A, 0x08, 0x0A, 0x88, 0x88, 0x70, 0x00,
	0x02, 0x01, 0x88, 0xA2, 0xC2, 0xC0, 0x90, 0x88, 0x88, 0x00,
	0x02, 0x00, 0x80, 0x82, 0x80, 0x80, 0x80, 0x80, 0xF8, 0x00,
	0x03, 0x01, 0x8A, 0xAA, 0x88, 0x8A, 0x88, 0x88, 0x88, 0x00,
	0x03, 0x00, 0x8A, 0xAA, 0x98, 0x8A, 0x88, 0x88, 0x88, 0x00,
	0x02, 0x00, 0x72, 0x8A, 0x88, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x02, 0x00, 0xF2, 0x8A, 0x88, 0xF0, 0x80, 0x80, 0x80, 0x00,
	0x02, 0x00, 0x72, 0x8A, 0x88, 0x8A, 0x88, 0x89, 0x70, 0x0C,
	0x02, 0x00, 0xF2, 0x8A, 0x8A, 0xF0, 0x90, 0x88, 0x88, 0x00,
	0x02, 0x00, 0x72, 0x80, 0x80, 0x72, 0x08, 0x88, 0x70, 0x00,
	0x00, 0x02, 0xF8, 0x20, 0x22, 0x20, 0x20, 0x20, 0x20, 0x00,
	0x02, 0x00, 0x8A, 0x8A, 0x88, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x02, 0x00, 0x8A, 0x89, 0x89, 0x50, 0x50, 0x20, 0x20, 0x00,
	0x02, 0x02, 0xAA, 0xAA, 0xAA, 0xAA, 0x50, 0x50, 0x50, 0x00,
	0x02, 0x00, 0x8A, 0x51, 0x51, 0x20, 0x50, 0x88, 0x88, 0x00,
	0x02, 0x00, 0x8A, 0x50, 0x52, 0x20, 0x20, 0x20, 0x20, 0x00,
	0x00, 0x00, 0xFA, 0x11, 0x10, 0x20, 0x40, 0x80, 0xF8, 0x00,
	0x01, 0x70, 0x40, 0x41, 0x40, 0x40, 0x41, 0x43, 0x40, 0x00,
	0x02, 0x00, 0x80, 0x40, 0x42, 0x20, 0x10, 0x10, 0x0A, 0x00,
	0x00, 0x71, 0x10, 0x10, 0x11, 0x10, 0x11, 0x13, 0x10, 0x00,
	0x02, 0x20, 0x52, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x03, 0x02, 0x00,
	0x00, 0x42, 0x40, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x01, 0x73, 0x0A, 0x88, 0x88, 0x78, 0x00,
	0x02, 0x00, 0x80, 0x82, 0xF0, 0x8A, 0x88, 0x88, 0xF0, 0x00,
	0x00, 0x00, 0x00, 0x02, 0x70, 0x88, 0x80, 0x88, 0x70, 0x00,
	0x00, 0x00, 0x0A, 0x0A, 0x78, 0x8A, 0x88, 0x88, 0x78, 0x00,
	0x00, 0x00, 0x00, 0x03, 0x73, 0x8A, 0x80, 0x88, 0x70, 0x00,
	0x00, 0x02, 0x18, 0x20, 0x22, 0xF8, 0x20, 0x20, 0x20, 0x00,
	0x00, 0x00, 0x00, 0x02, 0x78, 0x8A, 0x8A, 0x78, 0x0A, 0x70,
	0x02, 0x00, 0x80, 0x82, 0xB0, 0xCA, 0x88, 0x88, 0x88, 0x00,
	0x00, 0x20, 0x00, 0x00, 0x62, 0x20, 0x20, 0x20, 0x70, 0x00,
	0x00, 0x10, 0x00, 0x00, 0x31, 0x10, 0x10, 0x11, 0x10, 0xE0,
	0x02, 0x00, 0x80, 0x83, 0x90, 0xA0, 0xA0, 0x90, 0x88, 0x00,
	0x00, 0x02, 0x60, 0x20, 0x22, 0x20, 0x20, 0x20, 0x30, 0x00,
	0x00, 0x00, 0x00, 0x02, 0xF2, 0xAA, 0xA8, 0xA8, 0xA8, 0x00,
	0x00, 0x00, 0x00, 0x02, 0xB0, 0xCA, 0x88, 0x88, 0x88, 0x00,
	0x00, 0x00, 0x00, 0x02, 0x70, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x00, 0x00, 0x02, 0xF0, 0x8A, 0x8A, 0xF0, 0x80, 0x80,
	0x00, 0x00, 0x00, 0x02, 0x78, 0x8A, 0x88, 0x78, 0x0A, 0x08,
	0x00, 0x00, 0x00, 0x02, 0xB0, 0xC8, 0x80, 0x80, 0x80, 0x00,
	0x00, 0x00, 0x00, 0x01, 0x73, 0x88, 0x08, 0x88, 0x70, 0x00,
	0x01, 0x00, 0x40, 0x41, 0xF0, 0x40, 0x40, 0x48, 0x30, 0x00,
	0x00, 0x00, 0x00, 0x02, 0x88, 0x8A, 0x88, 0x98, 0x68, 0x00,
	0x00, 0x00, 0x00, 0x01, 0x89, 0x88, 0x50, 0x20, 0x20, 0x00,
	0x00, 0x00, 0x00, 0x02, 0xAA, 0xAA, 0xA8, 0x50, 0x50, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x8A, 0x50, 0x20, 0x50, 0x88, 0x00,
	0x00, 0x00, 0x00, 0x02, 0x88, 0x8A, 0x8A, 0x78, 0x0A, 0x70,
	0x00, 0x00, 0x00, 0x00, 0xFA, 0x10, 0x40, 0x80, 0xF8, 0x00,
	0x00, 0x1A, 0x20, 0x20, 0x22, 0xC0, 0x20, 0x21, 0x22, 0x00,
	0x00, 0x22, 0x20, 0x20, 0x22, 0x20, 0x20, 0x22, 0x20, 0x00,
	0x00, 0xC2, 0x20, 0x20, 0x22, 0x18, 0x23, 0x20, 0x20, 0x00,
	0x02, 0x03, 0x68, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x20, 0x00, 0x22, 0x20, 0x20, 0x20, 0x20, 0x00,
	0x00, 0x02, 0x20, 0x72, 0xAA, 0xA0, 0xA0, 0x78, 0x20, 0x00,
	0x01, 0x00, 0x32, 0x41, 0x40, 0xF0, 0x40, 0x60, 0x98, 0x00,
	0x00, 0x00, 0x00, 0x89, 0x71, 0x50, 0x70, 0x88, 0x00, 0x00,
	0x02, 0x00, 0x8A, 0x50, 0x52, 0xF8, 0xF8, 0x20, 0x20, 0x00,
	0x00, 0x22, 0x20, 0x20, 0x22, 0x00, 0x20, 0x22, 0x20, 0x00,
	0x02, 0x60, 0x90, 0x41, 0xA1, 0x90, 0x21, 0x12, 0x90, 0x00,
	0x00, 0x00, 0xD8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x02, 0x00, 0xFA, 0x71, 0x40, 0x40, 0x70, 0x88, 0xF8, 0x00,
	0x00, 0x00, 0x72, 0x78, 0x88, 0x78, 0xF8, 0x00, 0x00, 0x00,
	0x00, 0x01, 0x08, 0x29, 0x51, 0xA0, 0x28, 0x10, 0x08, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0xFA, 0x08, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x00, 0x00, 0x00, 0x00,
	0x02, 0x00, 0xFA, 0x71, 0x51, 0x60, 0x50, 0x88, 0xF8, 0x00,
	0x00, 0x00, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x02, 0x61, 0x90, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x02, 0x00, 0x20, 0xFA, 0x20, 0x00, 0x00, 0xF8, 0x00,
	0x00, 0x61, 0x90, 0x20, 0x40, 0xF0, 0x00, 0x00, 0x00, 0x00,
	0x01, 0xF2, 0x10, 0x10, 0x90, 0x60, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x10, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x02, 0x88, 0x8A, 0x8A, 0x98, 0xE8, 0x80,
	0x03, 0x02, 0x7A, 0xE9, 0xEA, 0xEA, 0x28, 0x28, 0x28, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x22, 0x70, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x10, 0x40,
	0x02, 0x22, 0x60, 0x20, 0x20, 0x20, 0x00, 0x00, 0x00, 0x00,
	0x02, 0x61, 0x90, 0x93, 0x63, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x01, 0x00, 0x80, 0xA1, 0x51, 0x28, 0xA0, 0x40, 0x80, 0x00,
	0x03, 0x00, 0x40, 0x40, 0x43, 0x50, 0x50, 0x78, 0x10, 0x00,
	0x03, 0x00, 0x40, 0x40, 0x42, 0x52, 0x08, 0x10, 0x38, 0x00,
	0x00, 0x02, 0xE0, 0x40, 0x23, 0xD0, 0x50, 0x78, 0x10, 0x00,
	0x00, 0x00, 0x20, 0x22, 0x20, 0x40, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x20, 0x10, 0x21, 0x21, 0x50, 0xF8, 0x88, 0x88, 0x00,
	0x00, 0x10, 0x20, 0x21, 0x21, 0x50, 0xF8, 0x88, 0x88, 0x00,
	0x00, 0x30, 0x48, 0x21, 0x21, 0x50, 0xF8, 0x88, 0x88, 0x00,
	0x00, 0x68, 0xB0, 0x21, 0x21, 0x50, 0xF8, 0x88, 0x88, 0x00,
	0x00, 0x50, 0x50, 0x21, 0x21, 0x50, 0xF8, 0x88, 0x88, 0x00,
	0x01, 0x30, 0x4A, 0x31, 0x21, 0x50, 0xF8, 0x88, 0x88, 0x00,
	0x02, 0x02, 0xF8, 0xA2, 0xA2, 0xF8, 0xA0, 0xA0, 0xB8, 0x00,
	0x02, 0x00, 0x72, 0x82, 0x80, 0x80, 0x80, 0x8A, 0x70, 0x40,
	0x00, 0x20, 0x10, 0xFB, 0x83, 0x80, 0x80, 0x80, 0xF8, 0x00,
	0x00, 0x10, 0x20, 0xFB, 0x83, 0x80, 0x80, 0x80, 0xF8, 0x00,
	0x00, 0x30, 0x48, 0xFB, 0x83, 0x80, 0x80, 0x80, 0xF8, 0x00,
	0x00, 0x50, 0x50, 0xFB, 0x83, 0x80, 0x80, 0x80, 0xF8, 0x00,
	0x00, 0x20, 0x10, 0x70, 0x22, 0x20, 0x20, 0x20, 0x70, 0x00,
	0x00, 0x10, 0x20, 0x70, 0x22, 0x20, 0x20, 0x20, 0x70, 0x00,
	0x00, 0x30, 0x48, 0x70, 0x22, 0x20, 0x20, 0x20, 0x70, 0x00,
	0x00, 0x50, 0x50, 0x70, 0x22, 0x20, 0x20, 0x20, 0x70, 0x00,
	0x01, 0x00, 0xF2, 0x49, 0x48, 0xEA, 0x48, 0x48, 0xF0, 0x00,
	0x00, 0x68, 0xB0, 0x8A, 0xC9, 0xAA, 0x88, 0x88, 0x88, 0x00,
	0x00, 0x20, 0x10, 0x72, 0x88, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x10, 0x20, 0x72, 0x88, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x30, 0x48, 0x72, 0x88, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x68, 0xB0, 0x72, 0x88, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x50, 0x50, 0x72, 0x88, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x00, 0x00, 0x89, 0x51, 0x20, 0x88, 0x00, 0x00, 0x00,
	0x02, 0x09, 0x72, 0xAA, 0xAA, 0xAA, 0xAA, 0xC8, 0x70, 0x00,
	0x00, 0x20, 0x10, 0x8A, 0x88, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x10, 0x20, 0x8A, 0x88, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x30, 0x48, 0x8A, 0x88, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x50, 0x50, 0x8A, 0x88, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x10, 0x20, 0x89, 0x89, 0x50, 0x20, 0x20, 0x20, 0x00,
	0x03, 0x83, 0x80, 0x8B, 0x8B, 0x88, 0x80, 0x80, 0x80, 0x00,
	0x01, 0x03, 0x00, 0x8A, 0x88, 0xF2, 0x8A, 0x88, 0xF0, 0x80,
	0x00, 0x20, 0x10, 0x01, 0x73, 0x0A, 0x88, 0x88, 0x78, 0x00,
	0x00, 0x10, 0x20, 0x01, 0x73, 0x0A, 0x88, 0x88, 0x78, 0x00,
	0x00, 0x30, 0x48, 0x01, 0x73, 0x0A, 0x88, 0x88, 0x78, 0x00,
	0x00, 0x68, 0xB0, 0x01, 0x73, 0x0A, 0x88, 0x88, 0x78, 0x00,
	0x00, 0x50, 0x50, 0x01, 0x73, 0x0A, 0x88, 0x88, 0x78, 0x00,
	0x00, 0x33, 0x48, 0x01, 0x73, 0x0A, 0x88, 0x88, 0x78, 0x00,
	0x00, 0x00, 0x00, 0x03, 0xFB, 0x2A, 0xA0, 0xA0, 0xF8, 0x00,
	0x00, 0x00, 0x00, 0x02, 0x70, 0x88, 0x80, 0x8A, 0x70, 0x40,
	0x00, 0x20, 0x10, 0x03, 0x73, 0x8A, 0x80, 0x88, 0x70, 0x00,
	0x00, 0x10, 0x20, 0x03, 0x73, 0x8A, 0x80, 0x88, 0x70, 0x00,
	0x00, 0x30, 0x48, 0x03, 0x73, 0x8A, 0x80, 0x88, 0x70, 0x00,
	0x00, 0x50, 0x50, 0x03, 0x73, 0x8A, 0x80, 0x88, 0x70, 0x00,
	0x00, 0x20, 0x10, 0x00, 0x62, 0x20, 0x20, 0x20, 0x70, 0x00,
	0x00, 0x10, 0x20, 0x00, 0x62, 0x20, 0x20, 0x20, 0x70, 0x00,
	0x00, 0x30, 0x48, 0x00, 0x62, 0x20, 0x20, 0x20, 0x70, 0x00,
	0x00, 0x50, 0x50, 0x00, 0x62, 0x20, 0x20, 0x20, 0x70, 0x00,
	0x03, 0x42, 0x38, 0x12, 0x70, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x68, 0xB0, 0x02, 0xB0, 0xCA, 0x88, 0x88, 0x88, 0x00,
	0x00, 0x20, 0x10, 0x02, 0x70, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x10, 0x20, 0x02, 0x70, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x30, 0x48, 0x02, 0x70, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x68, 0xB0, 0x02, 0x70, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x50, 0x50, 0x02, 0x70, 0x8A, 0x88, 0x88, 0x70, 0x00,
	0x00, 0x02, 0x00, 0x20, 0x00, 0xF8, 0x20, 0x20, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x0A, 0x72, 0x9A, 0xAA, 0xC8, 0x70, 0x00,
	0x00, 0x20, 0x10, 0x02, 0x88, 0x8A, 0x88, 0x98, 0x68, 0x00,
	0x00, 0x10, 0x20, 0x02, 0x88, 0x8A, 0x88, 0x98, 0x68, 0x00,
	0x00, 0x30, 0x48, 0x02, 0x88, 0x8A, 0x88, 0x98, 0x68, 0x00,
	0x00, 0x50, 0x50, 0x02, 0x88, 0x8A, 0x88, 0x98, 0x68, 0x00,
	0x00, 0x10, 0x20, 0x02, 0x88, 0x8A, 0x8A, 0x78, 0x0A, 0x70,
	0x02, 0x00, 0x00, 0x82, 0xF0, 0x8A, 0x8A, 0xF0, 0x80, 0x80,
	0x00, 0x50, 0x50, 0x02, 0x88, 0x8A, 0x8A, 0x78, 0x0A, 0x70,
};
#endif

static int
vramwrite_vga_putchar (unsigned char c)
{
	static int x, y;
	static u32 buf[16][8];
	int i, j, fontx, fonty, fontlen;
	u8 *p, b, *font;
	bool fontcompressed;

	font = vf.font;
	fontx = vf.fontx;
	fonty = vf.fonty;
	fontlen = vf.fontlen;
	fontcompressed = vf.fontcompressed;

	if (!font)
		return 0;
	if (!vga_is_ready ())
		return 0;
	switch (c) {
	case '\b':
		if (x)
			x--;
		else if (y)
			x = 79, y--;
		break;
	case '\r':
		x = 0;
		break;
	default:
		p = &font[c * fontlen];
		for (i = 0; i < fonty; i++) {
			b = *p++;
			for (j = 0; j < fontx; j++)
				buf[i][j] = ((b << j) & 0x80) ? 0xFF00FF00 : 0;
			if (fontcompressed) {
				j = (i & 3) << 1;
				buf[i | 3][j + 0] = (b & 2) ? 0xFF00FF00 : 0;
				buf[i | 3][j + 1] = (b & 1) ? 0xFF00FF00 : 0;
				if (i & 2)
					i++;
			}
		}
		vga_transfer_image (VGA_FUNC_TRANSFER_DIR_PUT, buf,
				    VGA_FUNC_IMAGE_TYPE_BGRX_8888,
				    sizeof buf[0], fontx, fonty, x * fontx,
				    y * fonty);
		x++;
		if (x <= 79)
			break;
		/* fall through */
	case '\n':
		x = 0;
		y++;
		if (y <= 24)
			break;
		for (i = 0; i < 24; i++) {
			vga_transfer_image (VGA_FUNC_TRANSFER_DIR_GET,
					    scrollbuf,
					    VGA_FUNC_IMAGE_TYPE_BGRX_8888,
					    sizeof scrollbuf[0] * fontx * 80,
					    fontx * 80, fonty, 0,
					    (i + 1) * fonty);
			vga_transfer_image (VGA_FUNC_TRANSFER_DIR_PUT,
					    scrollbuf,
					    VGA_FUNC_IMAGE_TYPE_BGRX_8888,
					    sizeof scrollbuf[0] * fontx * 80,
					    fontx * 80, fonty, 0, i * fonty);
		}
		scrollbuf[0] = 0;
		vga_fill_rect (scrollbuf, VGA_FUNC_IMAGE_TYPE_BGRX_8888,
			       0, 24 * fonty, fontx * 80, fonty);
		y = 24;
	}
	return 0;
}

void
vramwrite_putchar (unsigned char c)
{
	if (vramwrite_vga_putchar (c))
		return;
	vramwrite_arch_putchar (c);
}

static int
vramwrite_msghandler (int m, int c)
{
	if (m == 0)
		vramwrite_putchar (c);
	return 0;
}

static void
vramwrite_init_msg (void)
{
	msgregister ("vramwrite", vramwrite_msghandler);
}

void
vramwrite_init_global (void)
{
	vramwrite_arch_init_global ();
}

static void
vramwrite_init_bsp (void)
{
	bool tty_vga = false;

#ifdef TTY_VGA
	vf.font = mplus_f12r;
	vf.fontx = 6;
	vf.fonty = 13;
	vf.fontlen = 10;
	vf.fontcompressed = true;
	tty_vga = true;
#endif
	if (tty_vga) {
		vramwrite_arch_get_font (&vf);
		scrollbuf = alloc (80 * vf.fontx * vf.fonty *
				   sizeof scrollbuf[0]);
	}
}

INITFUNC ("msg0", vramwrite_init_msg);
INITFUNC ("bsp0", vramwrite_init_bsp);
