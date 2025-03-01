CONSTANTS-$(CONFIG_ENABLE_ASSERT) += -DENABLE_ASSERT
CONSTANTS-$(CONFIG_VTD_TRANS) += -DVTD_TRANS

CONSTANTS-$(CONFIG_SHADOW_UHCI) += -DSHADOW_UHCI
CONSTANTS-$(CONFIG_HANDLE_USBMSC) += -DHANDLE_USBMSC
CONSTANTS-$(CONFIG_HANDLE_USBHUB) += -DHANDLE_USBHUB
CONSTANTS-$(CONFIG_CONCEAL_USBCCID) += -DCONCEAL_USBCCID
CONSTANTS-$(CONFIG_PS2KBD_F10USB) += -DF10USBTEST
CONSTANTS-$(CONFIG_PS2KBD_F12USB) += -DF12DUMPEHCI
CONSTANTS-y += -DENABLE_DPRINTF

CFLAGS += -Idrivers/include
CXXFLAGS += -Idrivers/include
# CFLAGS += -Idrivers/tflite-micro
# CFLAGS += -Iscripts/keystroke_dynamics
# CFLAGS += -Idrivers/tflite-micro/third_party/flatbuffers/include
CXXFLAGS += -Idrivers/tflite-micro/third_party/flatbuffers/include
CXXFLAGS += -Idrivers/tflite-micro/third_party/gemmlowp
CXXFLAGS += -Iscripts/keystroke_dynamics
CXXFLAGS += -Idrivers/tflite-micro
# LDFLAGS += -lm
# LDFLAGS += -lstdc++
# CFLAGS += -Icrypto
CFLAGS += -msoft-float
OBJS += keystroke.o
OBJS += runtime.o
# CFLAGS += -Idrivers/tflite-micro/tensorflow
CXXFLAGS += -Idrivers/tflite-micro/tensorflow
# LDFLAGS += -Ldrivers/tflite-micro/gen/linux_x86_64_default_cmsis_nn_gcc/lib
# LDFLAGS += -L/usr/lib/gcc/$(shell gcc -dumpmachine)/$(shell gcc -dumpversion)
# LIBS += -lstdc++ -lsupc++ -lgcc_eh -lc++ drivers/tflite-micro/gen/linux_x86_64_default_cmsis_nn_gcc/lib/libtensorflow-microlite.a

objs-$(CONFIG_SHADOW_UHCI) += uhci.o uhci_debug.o uhci_shadow.o uhci_trans.o
objs-$(CONFIG_SHADOW_EHCI) += ehci.o ehci_debug.o ehci_shadow.o ehci_trans.o
objs-$(CONFIG_SHADOW_XHCI) += xhci.o xhci_shadow.o xhci_trans.o
objs-$(CONFIG_HANDLE_USBMSC) += usb_mscd.o
objs-$(CONFIG_HANDLE_USBHUB) += usb_hub.o
objs-$(CONFIG_CONCEAL_USBCCID) += usb_ccid.o
objs-y += usb.o usb_device.o usb_hid.o usb_hook.o usb_log.o keystroke.o
CXXFLAGS += -std=c++11
CXXFLAGS += $(WARNING_CXXFLAGS) -Idrivers/include
CXXFLAGS += -fno-exceptions -fno-rtti -fno-threadsafe-statics
CXXFLAGS += -fexceptions -frtti
# LIBS += -lstdc++ -lsupc++ -lgcc_eh

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

TFLM_PATH = drivers/tflite-micro

TFLM_LIB = $(TFLM_PATH)/gen/linux_x86_64_default_cmsis_nn_gcc/lib/libtensorflow-microlite.a
CXXFLAGS += -I$(TFLM_PATH)
CXXFLAGS += -I$(TFLM_PATH)/tensorflow/lite/micro
CXXFLAGS += -I$(TFLM_PATH)/tensorflow/lite/micro/kernels
CXXFLAGS += -I$(TFLM_PATH)/third_party/flatbuffers/include

LIBS += -Ldrivers/tflite-micro/gen/linux_x86_64_default_cmsis_nn_gcc/lib -ltensorflow-microlite
LDFLAGS += -Wl,--whole-archive -Wl,--no-undefined
LDFLAGS += -Ldrivers/tflite-micro/gen/linux_x86_64_default_cmsis_nn_gcc/lib
CXXFLAGS += -std=c++11 -fPIC
CXXFLAGS += -Wl,--no-undefined
CXXFLAGS += -fexceptions -frtti

drivers/usb/keystroke.o: drivers/usb/keystroke.cpp
	$(CXX) $(CXXFLAGS) $(CONSTANTS-y) -c $< -o $@

CFLAGS += -Idrivers/usb
