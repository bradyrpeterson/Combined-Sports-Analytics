import os

print("=" * 80)
print("DEBUGGING YOUR FILE STRUCTURE")
print("=" * 80)

# Check current directory
print(f"\n1. Current directory: {os.getcwd()}")

# Check if folders exist
print("\n2. Checking folders:")
for folder in ['football', 'basketball', 'templates', 'static']:
    exists = "✓ EXISTS" if os.path.exists(folder) else "✗ MISSING"
    print(f"   {folder}/  {exists}")

# Check football files
print("\n3. Football folder contents:")
if os.path.exists('football'):
    files = os.listdir('football')
    if files:
        for f in files:
            print(f"   - {f}")
    else:
        print("   (empty)")
else:
    print("   Football folder doesn't exist!")

# Check basketball files
print("\n4. Basketball folder contents:")
if os.path.exists('basketball'):
    files = os.listdir('basketball')
    if files:
        for f in files:
            print(f"   - {f}")
    else:
        print("   (empty)")
else:
    print("   Basketball folder doesn't exist!")

# Check for specific files
print("\n5. Looking for required JSON files:")
files_to_check = [
    'football/fbs_teams_2025.json',
    'football/team_logos.json',
    'football/team_color.json',
    'basketball/d1_teams_2025.json'
]
for filepath in files_to_check:
    exists = "✓ FOUND" if os.path.exists(filepath) else "✗ MISSING"
    print(f"   {filepath}  {exists}")

print("\n" + "=" * 80)
print("INSTRUCTIONS:")
print("=" * 80)
print("Run this script from your sports-analytics folder:")
print("  cd sports-analytics")
print("  python debug_script.py")
print("=" * 80)