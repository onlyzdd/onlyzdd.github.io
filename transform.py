from pathlib import Path


def process_front_matter(infp):
    front_matter = {}
    has_start_line = False
    for line in infp:
        if line.startswith("---"):
            if not has_start_line:
                has_start_line = True
                continue
            else:
                break
        if has_start_line:
            key, value = line.split(":", 1)
            front_matter[key.strip()] = value.strip()
    return front_matter


def process_body(infp):
    lines = []
    for line in infp:
        if line.startswith("{% raw %}") or line.startswith("{% endraw %}"):
            continue
        if line.startswith("{: .prompt-tip }"):
            line = "> [!TIP]\n"
        elif line.startswith("{: .prompt-info }"):
            line = "> [!NOTE]\n"
        elif line.startswith("{: .prompt-warning }"):
            line = "> [!WARNING]\n"
        elif line.startswith("{: .prompt-danger }"):
            line = "> [!CAUTION]\n"
        else:
            pass
        lines.append(line)
    return "".join(lines)


input_dir = Path("./_posts")
output_dir = Path("./docs")


for input_file in input_dir.glob("*.md"):
    output_file = output_dir / input_file.name
    if output_file.exists() and output_file.lstat().st_mtime > input_file.lstat().st_mtime:
        continue
    print(f"processing {input_file}")
    with input_file.open() as infp, output_file.open("w") as outfp:
        front_matter = process_front_matter(infp)
        body = process_body(infp)
        title = f"# {front_matter.get('title')}"
        output = title.strip() + "\n\n" + body.lstrip()
        outfp.write(output)
