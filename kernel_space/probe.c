#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

// Define the structure of the data we will send to Python
struct data_t {
    u32 pid;
    char comm[16];
};

// Create a perf ring buffer named 'socket_events'
BPF_PERF_OUTPUT(socket_events);

// Hook into the 'sys_enter_connect' syscall
TRACEPOINT_PROBE(syscalls, sys_enter_connect) {
    struct data_t data = {};
    
    // Get the Process ID (PID)
    data.pid = bpf_get_current_pid_tgid() >> 32;
    
    // Get the Command/Container name making the request
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    
    // Submit the event to the ring buffer
    socket_events.perf_submit(args, &data, sizeof(data));
    return 0;
}
