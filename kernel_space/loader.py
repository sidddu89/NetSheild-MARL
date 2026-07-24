#!/usr/bin/python3
from bcc import BPF

# 1. Load the C probe file
with open("kernel_space/probe.c", "r") as f:
    bpf_text = f.read()

# 2. Initialize BPF
b = BPF(text=bpf_text)

print("🛡️ NetShield eBPF Probe Active.")
print("Intercepting sys_enter_connect events... (Press Ctrl+C to stop)")
print(f"{'PID':<10} {'COMMAND':<20}")

# 3. Define the callback for the perf ring buffer
def print_event(cpu, data, size):
    event = b["socket_events"].event(data)
    print(f"{event.pid:<10} {event.comm.decode('utf-8', 'replace'):<20}")

# 4. Open the perf ring buffer
b["socket_events"].open_perf_buffer(print_event)

# 5. Poll the buffer indefinitely
while True:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        print("\nDetaching probe...")
        exit()
