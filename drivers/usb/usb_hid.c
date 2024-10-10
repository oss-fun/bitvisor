/*
 * Copyright (c) 2010 Igel Co., Ltd
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

#include <core.h>
#include "uhci.h"
#include "usb.h"
#include "usb_device.h"
#include "usb_hook.h"
#include "usb_log.h"

#define USB_ICLASS_HID  0x3
#define USB_PROTOCOL_KEYBOARD  0x1

static int
hid_intercept(struct usb_host *usbhc,
	      struct usb_request_block *urb, void *arg)
{
	struct usb_buffer_list *ub;
        for(ub = urb->shadow->buffers; ub; ub = ub->next){
            if (ub->pid != USB_PID_IN)
                continue;
            u8 *cp;
            cp = (u8 *)mapmem_as(as_passvm, ub->padr, ub->len, 0);
            for(int i = 0; i < ub->len; i++){
                if(i >= 2 && *cp == 0x1d){ // z
                    *cp = 0x04; // a
                }
                cp++;
            }
        }	
	return USB_HOOK_PASS;
}

/* CAUTION:
   This handler reads an interface descriptor, so it
   must be initialized *AFTER* the device management
   has been initialized.

   Currently the hook is created for the whole device,
   so every enpoint will be managed by the hook.
*/
void
usbhid_init_handle (struct usb_host *host, struct usb_device *dev)
{
	    u8 class, protocol;
        int i;
        struct usb_interface_descriptor *ides;

        if (!dev || !dev->config || !dev->config->interface ||
            !dev->config->interface->altsetting ||
            !dev->config->interface->num_altsetting) {
            dprintft(1, "HID(%02x): interface descriptor not found.\n",
                 dev->devnum);
            return;
        }
        for (i = 0; i < dev->config->interface->num_altsetting; i++) {
            ides = dev->config->interface->altsetting + i;
            class = ides->bInterfaceClass;
            protocol = ides->bInterfaceProtocol;
            if (class == USB_ICLASS_HID && protocol == USB_PROTOCOL_KEYBOARD)
                break;
        }

        if (i == dev->config->interface->num_altsetting)
            return;

        printf("HID(%02x): an USB keyboard found.\n", dev->devnum);

        spinlock_lock(&host->lock_hk);
        struct usb_endpoint_descriptor *epdesc;
        for(i = 1; i <= ides->bNumEndpoints; i++){
            epdesc = &ides->endpoint[i];
            if (epdesc->bEndpointAddress & USB_ENDPOINT_IN) {
                usb_hook_register(host, USB_HOOK_REPLY,
                          USB_HOOK_MATCH_DEV | USB_HOOK_MATCH_ENDP,
                          dev->devnum, epdesc->bEndpointAddress,
                          NULL, hid_intercept, dev, dev);
                printf("HID(%02x, %02x): HID device monitor registered.\n",
                        dev->devnum, epdesc->bEndpointAddress);
            }
        }
        spinlock_unlock(&host->lock_hk);

    dprintft(1, "HID(%02x): HID device monitor registered.\n",
		 dev->devnum);

	return;
}
