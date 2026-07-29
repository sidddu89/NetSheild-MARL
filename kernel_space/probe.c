#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/socket.h>
#include <net/sock.h>
#include <bcc/proto.h>

// Struct submitted to perf ring buffer for each connect syscall
struct conn_event_t {
    u32 pid;                // Process ID
    u32 tgid;               // Thread Group ID / Process Group
    u32 uid;                // User ID
    char comm[16];          // Process Command Name (e.g. wget, curl, python)
    u32 daddr;              // Destination IPv4 address (network byte order)
    u16 dport;              // Destination Port (network byte order)
    u16 family;             // AF_INET socket family
};

// Define BCC perf output buffer
BPF_PERF_OUTPUT(socket_events);

// Hook into sys_enter_connect syscall
TRACEPOINT_PROBE(syscalls, sys_enter_connect) {
    struct conn_event_t event = {};
    
    // Extract PID and TGID
    u64 pid_tgid = bpf_get_current_pid_tgid();
    event.pid = pid_tgid >> 32;
    event.tgid = (u32)pid_tgid;
    
    // Extract User ID
    event.uid = bpf_get_current_uid_gid();
    
    // Extract process command name
    bpf_get_current_comm(event.comm, sizeof(event.comm));
    
    // Read the sockaddr structure passed to sys_enter_connect
    struct sockaddr *addr = (struct sockaddr *)args->uservaddr;
    if (addr == NULL) {
        return 0;
    }
    
    // Read socket family safely from user space
    bpf_probe_read_user(&event.family, sizeof(event.family), &addr->sa_family);
    
    // Filter for IPv4 (AF_INET = 2)
    if (event.family == AF_INET) {
        struct sockaddr_in *addr_in = (struct sockaddr_in *)addr;
        bpf_probe_read_user(&event.daddr, sizeof(event.daddr), &addr_in->sin_addr.s_addr);
        bpf_probe_read_user(&event.dport, sizeof(event.dport), &addr_in->sin_port);
        
        // Submit telemetry event to perf ring buffer
        socket_events.perf_submit(args, &event, sizeof(event));
    }
    
    return 0;
}
