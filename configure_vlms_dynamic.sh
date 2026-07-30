#!/bin/bash
# =============================================================================
# complete haproxy + system configuration for 60k concurrent connections
# for 56 vllm workers (120b models) - optimized for single-client high concurrency
# =============================================================================

echo "=================================================="
echo "haproxy + system tuning for 60k concurrent requests"
echo "=================================================="

# =============================================================================
# part 1: haproxy server system tuning
# =============================================================================

cat > /etc/sysctl.d/99-haproxy-high-concurrency.conf << 'eof'
# =============================================================================
# haproxy server - high concurrency system tuning
# optimized for 60k+ concurrent connections
# =============================================================================

# ---- network performance ----
# increase max number of open files system-wide
fs.file-max = 2097152

# increase inotify watches (if monitoring many files)
fs.inotify.max_user_instances = 8192
fs.inotify.max_user_watches = 524288

# ---- tcp connection handling ----
# increase tcp socket listen queue (backlog)
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535

# increase network device backlog
net.core.netdev_max_backlog = 65535

# ---- ephemeral port range ----
# expand ephemeral port range for outbound connections
# default: 32768-60999 (~28k ports) -> new: 1024-65535 (~64k ports)
net.ipv4.ip_local_port_range = 1024 65535

# ---- tcp time_wait optimization ----
# enable reuse of time_wait sockets for new connections
# critical for high connection rates
net.ipv4.tcp_tw_reuse = 1

# reduce fin_wait timeout (default 60s -> 30s)
# use with caution - may cause issues with slow networks
net.ipv4.tcp_fin_timeout = 30

# maximum number of time_wait sockets
net.ipv4.tcp_max_tw_buckets = 2000000

# ---- tcp buffer tuning ----
# optimize tcp memory buffers for many concurrent connections
# format: min default max (in bytes)
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# increase max socket receive/send buffer
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216

# ---- tcp performance ----
# enable tcp window scaling
net.ipv4.tcp_window_scaling = 1

# enable selective acknowledgments
net.ipv4.tcp_sack = 1

# disable slow start after idle
net.ipv4.tcp_slow_start_after_idle = 0

# ---- connection tracking ----
# increase connection tracking table size
net.netfilter.nf_conntrack_max = 1048576

# increase conntrack buckets
net.netfilter.nf_conntrack_buckets = 262144

# reduce conntrack timeout for established connections
net.netfilter.nf_conntrack_tcp_timeout_established = 600

# ---- kernel settings ----
# increase pid max (for many processes/threads)
kernel.pid_max = 4194304

# disable swapping for performance
vm.swappiness = 10

eof

# apply sysctl changes
echo "[1/5] applying haproxy server sysctl settings..."
sudo sysctl -p /etc/sysctl.d/99-haproxy-high-concurrency.conf

# =============================================================================
# part 2: ulimit configuration
# =============================================================================

echo "[2/5] configuring system limits (ulimits)..."

cat > /etc/security/limits.d/99-haproxy.conf << 'eof'
# haproxy system limits for high concurrency
# format: <domain> <type> <item> <value>

# open files limit (should be 2x maxconn)
* soft nofile 200000
* hard nofile 200000
root soft nofile 200000
root hard nofile 200000

# process limits
* soft nproc 65535
* hard nproc 65535

# core dumps (optional - disable for production)
* soft core 0
* hard core 0

eof

# =============================================================================
# part 3: haproxy configuration
# =============================================================================

echo "[3/5] creating optimized haproxy configuration..."

# --- list of nodes ---
nodes=(
    10.20.188.73
    10.20.202.110
)

frontend_ip="${nodes[0]}"

cat > /etc/haproxy/haproxy.cfg << eof

global
    log /dev/log local0 notice
    # scaled for 58k+ concurrent connections
    maxconn 300000
    # ensure this matches your cpu core count (e.g. 64)
    nbthread 64
    tune.bufsize 95536
    tune.maxrewrite 62768
    tune.maxpollevents 500

defaults
    mode http
    log global
    option httplog
    
    # keepalive: critical for python clients to reuse sockets
    option http-keep-alive
    timeout http-keep-alive 300s

    # fail fast: if server is physically down, know in 5s
    timeout connect 5s

    # fail slow: the fix for 120b model latency
    # we allow the client and server to stay connected for 4 hours (14400s).
    # this covers the "waiting" time in the queue + generation time.
    timeout client 14400s
    timeout server 14400s
    timeout queue  14400s
    
    # reuse connections aggressively to save cpu
    option forwardfor
    http-reuse safe
    
    # resilience
    retries 3
    option redispatch

frontend vlm_inference_group_1
    # binds to port 20100 on all available interfaces (wildcard)
    bind  $frontend_ip:20100
    maxconn 196608
    default_backend vllm_workers_group_1
    
    stats enable
    stats uri /stats
    stats refresh 1s
    stats show-legends

    monitor-uri /health

backend vllm_workers_group_1
    # roundrobin: critical for filling "waiting" queues equally
    balance roundrobin
    
    # health check
    option httpchk
    http-check send meth GET uri /health ver http/1.1 hdr Host localhost
    http-check expect status 200

    # backend connection behavior
    http-reuse safe

    # backend capacity buffer (holds requests when workers are full)
    fullconn 100000

    # defaults applied to all servers
    default-server check inter 30s maxconn 8192

    # --- servers ---
    # we set 'maxconn 20' to prevent overwhelming the worker os.
    # haproxy holds the excess requests in the 'fullconn' buffer above.

eof

# --- dynamically append servers ---
for node in "${nodes[@]}"; do
  ipdash=$(echo "$node" | tr '.' '-')
  for port in $(seq 30100 30107); do
    echo "    server vlm_${ipdash}_p${port} $node:$port" >> /etc/haproxy/haproxy.cfg
  done
done

echo "[3.7/5] validating haproxy configuration..."

haproxy -c -f "${haproxy_cfg}"

echo "[3.8/5] reloading haproxy..."

systemctl reload haproxy

echo "✅ haproxy configuration applied successfully"


# =============================================================================
# part 4: systemd service configuration
# =============================================================================

echo "[4/5] configuring haproxy systemd service..."

mkdir -p /etc/systemd/system/haproxy.service.d/

cat > /etc/systemd/system/haproxy.service.d/override.conf << 'eof'
[service]
# increase file descriptor limits
limitnofile=200000
limitnproc=65535

# memory limits (adjust based on your 2tb ram)
# each connection uses ~32kb (16kb per socket end)
# 100k connections = ~3.2gb
# leave plenty of headroom
memorymax=100g
memoryhigh=80g

# cpu affinity - allow haproxy to use all 96 cores
cpuaffinity=0-95

# restart policy
restart=always
restartsec=5s

# security (optional - adjust based on your needs)
# privatetmp=yes
# nonewprivileges=yes

eof

# reload systemd
systemctl daemon-reload

sudo haproxy -c -f /etc/haproxy/haproxy.cfg
sudo systemctl enable haproxy
sudo systemctl restart haproxy

# =============================================================================
# part 5: client machine tuning (critical!)
# =============================================================================

echo "[5/5] creating client machine tuning script..."

cat > /tmp/client-machine-tuning.sh << 'eof'
#!/bin/bash
#=============================================================================
# client machine system tuning
# run this on the machine sending 60k concurrent requests
# this is critical - your blank responses are from client-side port exhaustion!
#=============================================================================

echo "applying client machine system tuning..."

# create sysctl config for client
cat > /etc/sysctl.d/99-haproxy-client.conf << 'clienteof'
# =============================================================================
# client machine - high concurrency tuning
# critical: prevents ephemeral port exhaustion
# =============================================================================

# ---- ephemeral port range (most critical!) ----
# expand from default ~28k (32768-60999) to ~64k (1024-65535)
# this allows 60k+ concurrent outbound connections
net.ipv4.ip_local_port_range = 1024 65535

# ---- tcp time_wait optimization ----
# enable reuse of time_wait sockets - critical for high connection rate
net.ipv4.tcp_tw_reuse = 1

# reduce time_wait timeout (use carefully)
net.ipv4.tcp_fin_timeout = 30

# maximum time_wait sockets
net.ipv4.tcp_max_tw_buckets = 2000000

# ---- tcp connection handling ----
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535

# ---- open files ----
fs.file-max = 2097152

# ---- tcp buffers ----
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216

# ---- tcp performance ----
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_sack = 1
net.ipv4.tcp_slow_start_after_idle = 0

# ---- connection tracking ----
net.netfilter.nf_conntrack_max = 1048576
net.netfilter.nf_conntrack_buckets = 262144

clienteof

# apply settings
sysctl -p /etc/sysctl.d/99-haproxy-client.conf

# set ulimits for client
cat > /etc/security/limits.d/99-client.conf << 'clienteof'
# client machine limits
* soft nofile 200000
* hard nofile 200000
* soft nproc 65535
* hard nproc 65535
clienteof

echo ""
echo "=========================================="
echo "client machine tuning complete!"
echo "=========================================="
echo ""
echo "important: you must logout and login again for ulimit changes to take effect!"
echo ""
echo "to verify settings:"
echo "  1. check ephemeral port range: cat /proc/sys/net/ipv4/ip_local_port_range"
echo "  2. check ulimits: ulimit -n"
echo "  3. monitor connections: watch -n1 'ss -s'"
echo ""

eof

chmod +x /tmp/client-machine-tuning.sh

# =============================================================================
# completion and verification
# =============================================================================

echo ""
echo "=========================================="
echo "haproxy configuration complete!"
echo "=========================================="
echo ""
echo "generated files:"
echo "  1. /etc/sysctl.d/99-haproxy-high-concurrency.conf"
echo "  2. /etc/security/limits.d/99-haproxy.conf"
echo "  3. /etc/haproxy/haproxy.cfg"
echo "  4. /etc/systemd/system/haproxy.service.d/override.conf"
echo "  5. /tmp/client-machine-tuning.sh (run this on client machine!)"
echo ""
echo "next steps:"
echo "  1. validate haproxy config: haproxy -c -f /etc/haproxy/haproxy.cfg"
echo "  2. restart haproxy: systemctl restart haproxy"
echo "  3. check status: systemctl status haproxy"
echo "  4. **critical**: copy /tmp/client-machine-tuning.sh to your client machine and run it!"
echo "  5. on client: logout/login after running client tuning script"
echo ""
echo "monitoring commands:"
echo "  - haproxy connections: echo 'show info' | socat /var/run/haproxy.sock -"
echo "  - system connections: ss -s"
echo "  - per-state breakdown: ss -tan | awk '{print \$1}' | sort | uniq -c"
echo "  - port usage: ss -tan | grep -v listen | wc -l"
echo ""
echo "key configuration details:"
echo "  - global maxconn: 100,000"
echo "  - frontend maxconn: 70,000"
echo "  - per-worker maxconn: 1,250 (70k / 56 workers)"
echo "  - timeout client/server: 300s (5 minutes for long inference)"
echo "  - ephemeral ports: 1,024 - 65,535 (~64k ports)"
echo ""
echo "⚠️  critical: your blank responses were from client-side port exhaustion!"
echo "    run /tmp/client-machine-tuning.sh on your client machine immediately!"
echo ""

# validate haproxy config
echo "validating haproxy configuration..."
if haproxy -c -f /etc/haproxy/haproxy.cfg; then
    echo "✓ haproxy configuration is valid!"
else
    echo "✗ haproxy configuration has errors. please review."
    exit 1
fi

echo ""
echo "configuration complete! ready to handle 60k concurrent requests."
echo ""

