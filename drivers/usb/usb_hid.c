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
#include <core/time.h>
#include "keystroke.h"

#define USB_ICLASS_HID  0x3
#define USB_PROTOCOL_KEYBOARD  0x1
#define MAX_KEYS 6
#define RECORD_START_MOD 0x03  // Ctrl(0x01) + Shift(0x02)
#define RECORD_START_KEY 0x3B  // F2
#define RECORD_STOP_MOD  0x03  // Ctrl + Shift
#define RECORD_STOP_KEY  0x3C  // F3

// Keystroke dynamics parameters
#define FEATURE_COUNT 30        // Number of keystroke features to track
#define MIN_KEYSTROKE_COUNT 10  // Minimum keystrokes before authentication
#define CONFIDENCE_THRESHOLD 750 // Authentication threshold (75.0% * 1000)

// Fixed point scaling (10^3 = 1000)
#define FIXED_POINT_SCALE 1000

static const char *hid_keycode_to_ascii[256] = {
    [0x04] = "a", [0x05] = "b", [0x06] = "c", [0x07] = "d", [0x08] = "e",
    [0x09] = "f", [0x0A] = "g", [0x0B] = "h", [0x0C] = "i", [0x0D] = "j",
    [0x0E] = "k", [0x0F] = "l", [0x10] = "m", [0x11] = "n", [0x12] = "o",
    [0x13] = "p", [0x14] = "q", [0x15] = "r", [0x16] = "s", [0x17] = "t",
    [0x18] = "u", [0x19] = "v", [0x1A] = "w", [0x1B] = "x", [0x1C] = "y",
    [0x1D] = "z",
    [0x1E] = "1", [0x1F] = "2", [0x20] = "3", [0x21] = "4", [0x22] = "5",
    [0x23] = "6", [0x24] = "7", [0x25] = "8", [0x26] = "9", [0x27] = "0",

    // ten key
    [0x59] = "1", [0x5A] = "2", [0x5B] = "3", [0x5C] = "4", [0x5D] = "5",
    [0x5E] = "6", [0x5F] = "7", [0x60] = "8", [0x61] = "9", [0x62] = "0"
};

const char *keycode_to_ascii(u8 keycode) {
    return hid_keycode_to_ascii[keycode];
}

static u8 modifiers;
static u32 press_time;
static long long last_key_up_time = 0;  // 最後にキーが離された時刻

struct key_state {
    bool is_pressed;
    u8 modifiers;
    u32 press_time;
};

struct keyboard_data {
    struct key_state key_states[256];
    bool is_authorized;
    int password_index;
    bool is_recording;
    int sample_count;
    long long session_start_time;
};

static struct key_state key_states[256] = {0};
static u8 current_keys[MAX_KEYS] = {0};
static bool is_authorized = false;
// static char* password = "password";
static int password_index = 0;
static int password_length = 10;

// Keystroke feature structure
struct keystroke_feature {
    u8 keycode;
    long long press_time;
    long long release_time;
    long long hold_time;
    long long flight_time;
};

// Recent keystroke features for authentication
static struct keystroke_feature recent_keystrokes[FEATURE_COUNT] = {0};
static int keystroke_index = 0;
static int input_features[FEATURE_COUNT] = {0};
static long long last_key_up_time = 0;

// Feature statistics for normalization
static struct {
    int hold_time_mean;
    int hold_time_std;
    int flight_time_mean;
    int flight_time_std;
    bool initialized;
} feature_stats = {
    .hold_time_mean = 100,  // 100ms average hold time
    .hold_time_std = 50,    // 50ms standard deviation
    .flight_time_mean = 150, // 150ms average flight time
    .flight_time_std = 80,   // 80ms standard deviation
    .initialized = true
};

// Normalize a feature value using z-score normalization (integer math)
static int normalize_feature(int value, int mean, int std_dev) {
    if (std_dev == 0) return 0;
    return ((value - mean) * FIXED_POINT_SCALE) / std_dev;
}

// Prepare input features for the model
static void prepare_features_for_model() {
    // Reset input features
    memset(input_features, 0, sizeof(input_features));

    int index = 0;

    // Process hold times (first half of features)
    for (int i = 0; i < keystroke_index && i < FEATURE_COUNT/2; i++) {
        if (index < FEATURE_COUNT) {
            // Normalize hold time and store it
            input_features[index++] = normalize_feature(
                (int)recent_keystrokes[i].hold_time,
                feature_stats.hold_time_mean,
                feature_stats.hold_time_std
            );
        }
    }

    // Process flight times (second half of features)
    for (int i = 1; i < keystroke_index && i < FEATURE_COUNT/2; i++) {
        if (index < FEATURE_COUNT) {
            // Normalize flight time and store it
            input_features[index++] = normalize_feature(
                (int)recent_keystrokes[i].flight_time,
                feature_stats.flight_time_mean,
                feature_stats.flight_time_std
            );
        }
    }

    // Pad remaining features with zeros if we don't have enough keystrokes
    while (index < FEATURE_COUNT) {
        input_features[index++] = 0;
    }
}

// Calculate confidence score using the model
static int calculate_confidence_score() {
    // Prepare features for the model
    prepare_features_for_model();

    // Call model inference function from keystroke.h
    int confidence = keystroke_predict(input_features, FEATURE_COUNT);

    // Debug output
    printf("Model inference result: %d\n", confidence);

    return confidence;
}

// Check if the keystroke pattern matches the authorized user
static bool verify_keystroke_pattern() {
    if (keystroke_index < MIN_KEYSTROKE_COUNT) {
        printf("Not enough keystrokes for authentication (%d/%d)\n",
               keystroke_index, MIN_KEYSTROKE_COUNT);
        return false;
    }

    int confidence = calculate_confidence_score();
    bool authenticated = confidence > CONFIDENCE_THRESHOLD;

    printf("Authentication result: %s (confidence: %d.%d, threshold: %d.%d)\n",
           authenticated ? "AUTHENTICATED" : "REJECTED",
           confidence / FIXED_POINT_SCALE,
           (confidence % FIXED_POINT_SCALE) / (FIXED_POINT_SCALE / 100),
           CONFIDENCE_THRESHOLD / FIXED_POINT_SCALE,
           (CONFIDENCE_THRESHOLD % FIXED_POINT_SCALE) / (FIXED_POINT_SCALE / 100));

    return authenticated;
}

// Update the keystroke timing database with a new keystroke
static void add_keystroke_feature(struct keyboard_data *kbd_data, u8 keycode,
                                 long long press_time, long long release_time) {
    // Shift features if buffer is full
    if (keystroke_index >= FEATURE_COUNT) {
        // Move all elements one position left
        for (int i = 0; i < FEATURE_COUNT - 1; i++) {
            recent_keystrokes[i] = recent_keystrokes[i + 1];
        }
        keystroke_index = FEATURE_COUNT - 1;
    }

    // Add new keystroke data
    recent_keystrokes[keystroke_index].keycode = keycode;
    recent_keystrokes[keystroke_index].press_time = press_time;
    recent_keystrokes[keystroke_index].release_time = release_time;
    recent_keystrokes[keystroke_index].hold_time = release_time - press_time;

    // Calculate flight time if not the first keystroke
    if (keystroke_index > 0) {
        recent_keystrokes[keystroke_index].flight_time =
            press_time - recent_keystrokes[keystroke_index - 1].release_time;
    } else {
        recent_keystrokes[keystroke_index].flight_time = 0;
    }

    // Debug output
    printf("Added keystroke: key=%d, hold=%lldms, flight=%lldms\n",
           keycode, recent_keystrokes[keystroke_index].hold_time,
           recent_keystrokes[keystroke_index].flight_time);

    keystroke_index++;

    // Try authentication when enough keystrokes are collected
    if (keystroke_index >= MIN_KEYSTROKE_COUNT && !kbd_data->is_authorized) {
        bool is_authenticated = verify_keystroke_pattern();
        if (is_authenticated) {
            kbd_data->is_authorized = true;
            printf("User authenticated by keystroke dynamics!\n");
        }
    }
}

static void start_recording(struct keyboard_data *kbd_data, long long current_time_us) {
    if (!kbd_data->is_recording) {
        kbd_data->is_recording = true;
        kbd_data->session_start_time = current_time_us;
        printf("Recording started at time: %lld\n", current_time_us);
		printf("EVENT,KEY,TIMESTAMP,TIMING\n");
    }
}

static void stop_recording(struct keyboard_data *kbd_data) {
    if (kbd_data->is_recording) {
        kbd_data->is_recording = false;
        printf("Recording stopped. Total samples: %d\n", kbd_data->sample_count);
    }
}

static char*
unixtime_to_date(long long second, char* buf) {
	// UTC -> JST
    second += 9 * 60 * 60;

	// seconds_per_day
    const int seconds_per_day = 24 * 60 * 60;

    long days = (long)second / seconds_per_day;

    int year = 1970;
    int days_in_year;

    while (days > 0) {
        days_in_year = 365;
        if ((year % 4 == 0 && year % 100 != 0) || year % 400 == 0) {
            days_in_year = 366;
        }

        if (days >= days_in_year) {
            days -= days_in_year;
            year++;
        } else {
            break;
        }
    }

    int month = 1;
    int days_in_month[] = {31,28,31,30,31,30,31,31,30,31,30,31};

    if (days_in_year == 366) {
        days_in_month[1] = 29;
    }

    while (days >= days_in_month[month-1]) {
        days -= days_in_month[month-1];
        month++;
    }

	long hour = ((long)second % seconds_per_day) / 3600;

    int day = days + 1;

    buf[0] = '0' + (year / 1000);
    buf[1] = '0' + ((year / 100) % 10);
    buf[2] = '0' + ((year / 10) % 10);
    buf[3] = '0' + (year % 10);
    buf[4] = '0' + (month / 10);
    buf[5] = '0' + (month % 10);
    buf[6] = '0' + (day / 10);
    buf[7] = '0' + (day % 10);
	buf[8] = '0' + (hour / 10);
	buf[9] = '0' + (hour % 10);

    return buf;
}

static int
hid_intercept(struct usb_host *usbhc,
	      struct usb_request_block *urb, void *arg)
{
	long long second;
	int microsecond;
	get_epoch_time(&second, &microsecond);
	long long current_time_us = second * 1000000LL + microsecond;

	char buf[10];
	char* password = unixtime_to_date(second, buf);
	password[10] = '\0';

    struct keyboard_data *kbd_data = (struct keyboard_data *)arg;
    struct usb_buffer_list *ub;

    for(ub = urb->shadow->buffers; ub; ub = ub->next) {
        if (ub->pid != USB_PID_IN)
            continue;

        u8 *cp = (u8 *)mapmem_as(as_passvm, ub->padr, ub->len, 0);
        if (!cp || ub->len < 8) {
            if (cp) unmapmem(cp, ub->len);
            continue;
        }

        u8 modifiers = cp[0];

        bool current_pressed[256] = {false};
        for(int i = 2; i < 8; i++) {
            if (cp[i] != 0) {
                current_pressed[cp[i]] = true;
            }
        }

        for(int keycode = 0; keycode < 256; keycode++) {
            if (current_pressed[keycode]) {
				if (!kbd_data->key_states[keycode].is_pressed) {
					kbd_data->key_states[keycode].is_pressed = true;
					kbd_data->key_states[keycode].modifiers = modifiers;
					kbd_data->key_states[keycode].press_time = current_time_us;

					// Flight Time計算 (前のキーが離されてから今のキーが押されるまでの時間)
					long long flight_time = (last_key_up_time > 0) ? (current_time_us - last_key_up_time) : 0;

					if (kbd_data->is_recording) {
						kbd_data->sample_count++;
						printf("DOWN,KEY=0x%02x(%s),TIME=%lld,FLIGHT=%lld\n",
							keycode,
							hid_keycode_to_ascii[keycode] ? hid_keycode_to_ascii[keycode] : "?",
							current_time_us,
							flight_time);
					}


					// 録画開始コマンドの検出
					if ((modifiers & RECORD_START_MOD) == RECORD_START_MOD &&
						keycode == RECORD_START_KEY) {
						start_recording(kbd_data, current_time_us);
						continue;
					}

					// 録画停止コマンドの検出
					if ((modifiers & RECORD_STOP_MOD) == RECORD_STOP_MOD &&
						keycode == RECORD_STOP_KEY) {
						stop_recording(kbd_data);
						continue;
					}
				}
			} else {
				if (kbd_data->key_states[keycode].is_pressed) {
					// Hold Time計算 (キーが押されてから離されるまでの時間)
					long long hold_time = current_time_us - kbd_data->key_states[keycode].press_time;
					last_key_up_time = current_time_us;

					if (kbd_data->is_recording) {
						printf("UP,KEY=0x%02x(%s),TIME=%lld,HOLD=%lld\n",
							keycode,
							hid_keycode_to_ascii[keycode] ? hid_keycode_to_ascii[keycode] : "?",
							current_time_us,
							hold_time);
					}
                    // Add to keystroke features for authentication
                    add_keystroke_feature(
                        kbd_data,
                        keycode,
                        kbd_data->key_states[keycode].press_time,
                        current_time_us
                    );

                    kbd_data->key_states[keycode].is_pressed = false;
                    // const char *ascii = hid_keycode_to_ascii[keycode];
                    // if (ascii) {
                    //     if (!kbd_data->is_authorized) {
                    //         if (ascii[0] == password[kbd_data->password_index]) {
                    //             kbd_data->password_index++;
                    //             if (kbd_data->password_index == password_length) {
                    //                 kbd_data->is_authorized = true;
                    //                 printf("Authorized\n");
                    //             }
                    //         } else {
                    //             kbd_data->password_index = 0;
                    //         }
                    //     }
                    // }
				}
			}
        }

        // if (!kbd_data->is_authorized) {
        //     printf("Unauthorized\n");
        //     memset(cp, 0, ub->len);
        // }

        unmapmem(cp, ub->len);
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

		struct keyboard_data *kbd_data = alloc(sizeof(struct keyboard_data));
		if (!kbd_data) {
			printf("Failed to allocate keyboard data\n");
			return;
		}
		memset(kbd_data, -1, sizeof(struct keyboard_data));
		kbd_data->is_authorized = false;
		kbd_data->password_index = 0;
		kbd_data->is_recording = false;
		kbd_data->sample_count = 0;
		kbd_data->session_start_time = 0;

		// key_statesの初期化
		for (int i = 0; i < 256; i++) {
			kbd_data->key_states[i].is_pressed = false;
			kbd_data->key_states[i].modifiers = 0;
			kbd_data->key_states[i].press_time = 0;
		}

		// Initialize the keystroke dynamics model
		if (!keystroke_init()) {
			printf("Failed to initialize keystroke dynamics model\n");
		} else {
			printf("Keystroke dynamics model initialized successfully\n");
		}

        spinlock_lock(&host->lock_hk);
        struct usb_endpoint_descriptor *epdesc;
        for(i = 1; i <= ides->bNumEndpoints; i++){
            epdesc = &ides->endpoint[i];
            if (epdesc->bEndpointAddress & USB_ENDPOINT_IN) {
                usb_hook_register(host, USB_HOOK_REPLY,
                          USB_HOOK_MATCH_DEV | USB_HOOK_MATCH_ENDP,
                          dev->devnum, epdesc->bEndpointAddress,
                          NULL, hid_intercept, kbd_data, dev);
                printf("HID(%02x, %02x): HID device monitor registered.\n",
                        dev->devnum, epdesc->bEndpointAddress);
            }
        }
        spinlock_unlock(&host->lock_hk);

    dprintft(1, "HID(%02x): HID device monitor registered.\n",
		 dev->devnum);

	return;
}
