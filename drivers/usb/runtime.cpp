#include <cstdarg>
#include <cstdio>

extern "C" {
    int MicroPrintf(const char* format, ...) {
        va_list args;
        va_start(args, format);
        int result = vprintf(format, args);
        va_end(args);
        return result;
    }

    void* __dso_handle = 0;

    void __cxa_pure_virtual() {
        while(1);
    }

    int __cxa_atexit(void (*func)(void*), void* arg, void* dso) {
        return 0;
    }
}
