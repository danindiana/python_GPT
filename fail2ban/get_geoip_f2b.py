import subprocess
import requests
import json

# Function to get list of IPs from f2b_permanent set
def get_banned_ips():
    result = subprocess.run(['sudo', 'ipset', 'list', 'f2b_permanent'], capture_output=True, text=True)
    banned_ips = []
    for line in result.stdout.splitlines():
        if line and line[0].isdigit():  # Check if the line starts with a digit (an IP)
            banned_ips.append(line.strip())
    return banned_ips

# Function to get GeoIP info for a given IP address
def get_geoip_info(ip):
    try:
        # Using the ipinfo.io API (you could replace this with another API)
        response = requests.get(f'http://ip-api.com/json/{ip}')
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching GeoIP info for {ip}")
            return None
    except Exception as e:
        print(f"Failed to get GeoIP for {ip}: {e}")
        return None

# Main script
def main():
    banned_ips = get_banned_ips()
    print(f"Found {len(banned_ips)} banned IP(s) in f2b_permanent.\n")

    for ip in banned_ips:
        geoip_info = get_geoip_info(ip)
        if geoip_info:
            # Display the information (customize as needed)
            print(f"IP: {ip}")
            print(f"Country: {geoip_info.get('country')}")
            print(f"Region: {geoip_info.get('regionName')}")
            print(f"City: {geoip_info.get('city')}")
            print(f"ISP: {geoip_info.get('isp')}")
            print("---------")

if __name__ == "__main__":
    main()
