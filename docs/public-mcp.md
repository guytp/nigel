# Public MCP endpoint via Cloudflare

Open Nigel to the internet so claude.ai (or any web MCP client) can reach the MCP server without being on the LAN.

## Quick tunnel (no Cloudflare account)

```bash
ssh pi@nigel.local
cd /home/pi/nigel
./deploy/cloudflare-tunnel.sh
```

Installs `cloudflared` from Cloudflare's apt repo, drops a systemd unit (`picrawler-tunnel.service`), starts it, and prints the random `https://*.trycloudflare.com` URL when ready.

Re-check the URL later:

```bash
sudo journalctl -u picrawler-tunnel --since '1 hour ago' | grep trycloudflare | tail -1
```

The URL changes every restart. Fine for development.

## Connect claude.ai

In claude.ai Settings → Connectors → Add MCP server:

- **Name:** `nigel`
- **URL:** `https://<random-words>.trycloudflare.com/mcp`
- **Authentication:** if you set `MCP_TOKEN` in `/etc/picrawler-mcp.env`, add header `Authorization: Bearer <token>`

**Strongly set `MCP_TOKEN`** before enabling the tunnel — without it, anyone who learns the URL can drive Nigel's servos:

```bash
sudo tee /etc/picrawler-mcp.env > /dev/null <<EOF
OPENAI_API_KEY=sk-...
MCP_TOKEN=$(openssl rand -hex 32)
EOF
sudo chmod 600 /etc/picrawler-mcp.env
sudo systemctl restart picrawler-mcp
sudo grep MCP_TOKEN /etc/picrawler-mcp.env  # to copy into claude.ai
```

## Stable named tunnel (optional, requires Cloudflare account)

Quick tunnels are ephemeral. For a fixed subdomain like `nigel.yourdomain.com`, use a named tunnel:

```bash
cloudflared tunnel login        # browser pops, sign in to Cloudflare
cloudflared tunnel create nigel
cloudflared tunnel route dns nigel nigel.yourdomain.com
# Create ~/.cloudflared/config.yml with the tunnel UUID + ingress rules
sudo systemctl disable picrawler-tunnel        # stop the quick-tunnel unit
sudo cloudflared service install                # replaces with named-tunnel unit
```

## Uninstall

```bash
./deploy/cloudflare-tunnel.sh --uninstall
```

Stops and removes the systemd unit. `cloudflared` binary is left in place — remove with `sudo apt-get remove cloudflared` if desired.

## Security reminders

- **Always set `MCP_TOKEN`** before exposing the tunnel publicly. Servo control is a physical-world blast radius.
- `MCP_ALLOWED_HOSTS` can additionally restrict which Host headers are accepted (set to the tunnel hostname once it's stable).
- Cloudflare quick tunnels log every request on their side. Good for audit, bad if you consider that a leak.
