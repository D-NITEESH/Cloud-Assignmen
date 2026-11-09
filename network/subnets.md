# Task 5: Subnetting and Network Access Control

## 1. Subnet Allocation for 10.20.0.0/16

We are given the address block `10.20.0.0/16`. Subnets are allocated from the largest host requirement to the smallest, placed contiguously starting at `10.20.0.0`, using the smallest prefix length that accommodates each requirement.

The formula for usable hosts is:
$$\text{Usable Hosts} = 2^{32 - \text{prefix}} - 2$$

### Allocation Working

1. **`app` subnet (Requirement: 1,000 hosts)**
   - Need $2^h - 2 \ge 1000 \implies 2^h \ge 1002 \implies h = 10$ ($2^{10} = 1024$).
   - Prefix length: $32 - 10 = /22$.
   - Network address: `10.20.0.0/22`.
   - Host range: `10.20.0.1` to `10.20.3.254`.
   - Broadcast address: `10.20.3.255`.
   - Usable hosts: $1024 - 2 = 1022$.

2. **`public` subnet (Requirement: 250 hosts)**
   - Begins immediately after `app`: `10.20.4.0`.
   - Need $2^h - 2 \ge 250 \implies 2^h \ge 252 \implies h = 8$ ($2^8 = 256$).
   - Prefix length: $32 - 8 = /24$.
   - Network address: `10.20.4.0/24`.
   - Host range: `10.20.4.1` to `10.20.4.254`.
   - Broadcast address: `10.20.4.255`.
   - Usable hosts: $256 - 2 = 254$.

3. **`data` subnet (Requirement: 100 hosts)**
   - Begins immediately after `public`: `10.20.5.0`.
   - Need $2^h - 2 \ge 100 \implies 2^h \ge 102 \implies h = 7$ ($2^7 = 128$).
   - Prefix length: $32 - 7 = /25$.
   - Network address: `10.20.5.0/25`.
   - Host range: `10.20.5.1` to `10.20.5.126`.
   - Broadcast address: `10.20.5.127`.
   - Usable hosts: $128 - 2 = 126$.

4. **`management` subnet (Requirement: 20 hosts)**
   - Begins immediately after `data`: `10.20.5.128`.
   - Need $2^h - 2 \ge 20 \implies 2^h \ge 22 \implies h = 5$ ($2^5 = 32$).
   - Prefix length: $32 - 5 = /27$.
   - Network address: `10.20.5.128/27`.
   - Host range: `10.20.5.129` to `10.20.5.158`.
   - Broadcast address: `10.20.5.159`.
   - Usable hosts: $32 - 2 = 30$.

### Subnet Allocation Table

| Subnet Name | Network Address | Prefix | Usable Host Range | Broadcast Address | Usable Hosts |
|---|---|---|---|---|---|
| `app` | `10.20.0.0` | `/22` | `10.20.0.1` – `10.20.3.254` | `10.20.3.255` | 1,022 |
| `public` | `10.20.4.0` | `/24` | `10.20.4.1` – `10.20.4.254` | `10.20.4.255` | 254 |
| `data` | `10.20.5.0` | `/25` | `10.20.5.1` – `10.20.5.126` | `10.20.5.127` | 126 |
| `management` | `10.20.5.128` | `/27` | `10.20.5.129` – `10.20.5.158` | `10.20.5.159` | 30 |

Remaining unallocated space in `10.20.0.0/16`: `10.20.5.160` to `10.20.255.255`.

---

## 2. Network Access Control List (NACL) for Public Subnet

The Load Balancer lives in the `public` subnet (`10.20.4.0/24`).

### Inbound Rules

| Rule | Protocol | Port Range | Source CIDR | Action | Explanation |
|---|---|---|---|---|---|
| 50 | TCP | 0–65535 | `198.51.100.0/24` | `deny` | Explicit block for known abusive range before any allow rules |
| 100 | TCP | 80 | `0.0.0.0/0` | `allow` | Allow HTTP traffic from anywhere |
| 110 | TCP | 443 | `0.0.0.0/0` | `allow` | Allow HTTPS traffic from anywhere |
| 120 | TCP | 22 | `10.20.5.128/27` | `allow` | Allow SSH only from management subnet |
| 130 | TCP | 1024–65535 | `10.20.0.0/22` | `allow` | Allow backend reply traffic returning from app subnet |
| * | All | All | `0.0.0.0/0` | `deny` | Default deny for unmatched packets |

### Outbound Rules

| Rule | Protocol | Port Range | Destination CIDR | Action | Explanation |
|---|---|---|---|---|---|
| 100 | TCP | 8000 | `10.20.0.0/22` | `allow` | Forward requests to app services on port 8000 |
| 110 | TCP | 1024–65535 | `0.0.0.0/0` | `allow` | Return replies to client ephemeral ports |
| * | All | All | `0.0.0.0/0` | `deny` | Default deny for unmatched packets |

---

## 3. Explanations

### Why a Stateless NACL Needs Separate Rules for Reply Traffic
Unlike stateful firewalls (or Security Groups in AWS) that maintain connection state tables and automatically permit response packets for established sessions, a **stateless NACL** inspects each packet in complete isolation without session context. 

When a client initiates an inbound TCP connection to port 443, the inbound packet is evaluated against inbound rules. When the load balancer replies, the response packet travels outbound toward the client's randomly chosen ephemeral port (e.g., 1024–65535). Because the NACL has no memory of the initial handshake, an outbound rule explicitly permitting destination ports 1024–65535 to `0.0.0.0/0` is strictly necessary; without it, the response packet is discarded by the default deny rule. Similarly, when the load balancer makes an outbound request to port 8000 on the app subnet, the app subnet's return packet enters inbound on an ephemeral port, requiring rule 130.

### Why Rule Order Matters for the 198.51.100.0/24 Block
NACL rules are evaluated sequentially from the **lowest rule number upward**, and the **first matching rule terminates evaluation**. 

If the deny rule for `198.51.100.0/24` were assigned a rule number higher than the allow rule for port 443 (e.g., Rule 200 after Rule 110), an incoming packet from abusive IP `198.51.100.9` targeting port 443 would match Rule 110 (`0.0.0.0/0`, port 443, `allow`) first. The firewall would immediately permit the packet, completely bypassing the deny rule. Placing the deny rule at **Rule 50** ensures it is evaluated before any broad wildcard (`0.0.0.0/0`) allow rules, guaranteeing malicious traffic is dropped regardless of destination port.
