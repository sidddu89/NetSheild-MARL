#!/usr/bin/python3
import os
import sys
import time
import socket
import struct
# pyrefly: ignore [missing-import]
from bcc import BPF           

def check_root_privileges():
    """Ensure script is executed with root/sudo privileges on WSL2/Linux."""
    if os.geteuid() != 0:
        print("❌ ERROR: NetShield eBPF loader requires root privileges!")
        print("Please rerun with sudo: 'sudo python3 kernel_space/loader.py'")
        sys.exit(1)

class EBPFLoader:
    def __init__(self, probe_path=None, callback_fn=None):
        check_root_privileges()
        
        if probe_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            probe_path = os.path.join(base_dir, "probe.c")
            
        if not os.path.exists(probe_path):
            raise FileNotFoundError(f"Probe file not found at: {probe_path}")
            
        with open(probe_path, "r") as f:
            bpf_text = f.read()
            
        print("⚡ Compiling and loading eBPF C probe into kernel...")
        self.bpf = BPF(text=bpf_text)
        self.callback_fn = callback_fn
        
    def _parse_ip(self, ip_int):
        """Converts integer network byte order IP to standard IPv4 string."""
        return socket.inet_ntoa(struct.pack("<I", ip_int))

    def _parse_port(self, port_int):
        """Converts network byte order port to host integer."""
        return socket.ntohs(port_int)

    def _event_handler(self, cpu, data, size):
        event = self.bpf["socket_events"].event(data)
        comm_str = event.comm.decode('utf-8', errors='ignore').rstrip('\x00')
        dest_ip = self._parse_ip(event.daddr)
        dest_port = self._parse_port(event.dport)
        
        payload = {
            "pid": event.pid,
            "tgid": event.tgid,
            "uid": event.uid,
            "comm": comm_str,
            "dest_ip": dest_ip,
            "dest_port": dest_port
        }
        
        if self.callback_fn:
            self.callback_fn(payload)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] PID: {payload['pid']:<6} COMM: {payload['comm']:<15} DEST: {payload['dest_ip']}:{payload['dest_port']}")

    def start_listener(self):
        """Opens perf ring buffer and polls events efficiently."""
        self.bpf["socket_events"].open_perf_buffer(self._event_handler)
        print("🛡️ NetShield eBPF Probe Active. Intercepting sys_enter_connect...")
        print(f"{'TIMESTAMP':<10} {'PID':<6} {'COMMAND':<15} {'DESTINATION':<20}")
        print("-" * 55)
        
        try:
            while True:
                # Poll with 100ms timeout to prevent 100% CPU utilization
                self.bpf.perf_buffer_poll(timeout=100)
        except KeyboardInterrupt:
            print("\n🛑 Detaching eBPF probe from kernel. Exiting safely.")
            sys.exit(0)

if __name__ == "__main__":
    loader = EBPFLoader()
    loader.start_listener()
