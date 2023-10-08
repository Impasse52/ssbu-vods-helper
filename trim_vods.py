import os
import re
import sys


def crop(start, end, input, output) -> None:
    os.system(rf'ffmpeg -i "{input}" -ss {start} -to {end} -c copy "{output}"')


def tokenize_titles(filename: str) -> list:
    with open(filename) as vod_file:
        vod_file = "".join(vod_file.readlines())

    title_re = r"(.*) \| (.*) - (.*) \((.*)\) vs (.*) \((.*)\) -- (\d+:\d\d:\d\d) - (\d+:\d\d:\d\d)"
    title = re.findall(title_re, vod_file)
    print(title)

    return title


def trim_vods(titles: list, input_filename: str) -> None:
    for t in titles:
        output_filename = f"{(t[2] + t[4]).lower()}".replace(" ", "")
        print(output_filename)

        n = os.listdir().count(f"{output_filename}.mp4")

        if n > 0:
            output_filename = f"{output_filename}{n + 1}"

        start_time = t[6]
        end_time = t[7]

        crop(start_time, end_time, input_filename, f"{output_filename}.mp4")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        vod_dir = sys.argv[1]
        timestamps_dir = sys.argv[2]

        titles = tokenize_titles(timestamps_dir)
        trim_vods(titles, vod_dir)
    else:
        vod_dir = ""
        timestamps_dir = ""

        titles = tokenize_titles(timestamps_dir)
        trim_vods(titles, vod_dir)
