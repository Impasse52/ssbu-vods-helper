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


def trim_vods(titles: list, input_filename: str, videos_dir: str) -> None:
    for t in titles:
        output_filename = f"{(t[2] + t[4]).lower()}".replace(" ", "")
        print(output_filename)

        n = os.listdir().count(f"{output_filename}.mp4")

        if n > 0:
            output_filename = f"{output_filename}{n + 1}"

        start_time = t[6]
        end_time = t[7]

        crop(start_time, end_time, input_filename, f"{videos_dir}/{output_filename}.mp4")


if __name__ == "__main__":
    from pathlib import Path
    from thumbfair.thumbfair import gen_thumbnail_for_vod

    if len(sys.argv) > 1:
        vod_dir = sys.argv[1]
        timestamps_dir = sys.argv[2]
    else:
        vod_dir = "/mnt/Main/Documents/Misc/Syncthing/Code/Projects/VOD-Assistant/2023-10-15_1951643050_smashbrositalia_altomare_z_la_scomparsa_di_chrono_ft_forze_sasyzza_zio_al_and_more.mkv"
        timestamps_dir = "/home/impasse/Code/Projects/Thumbfair/AMSZ4.md"
        
    output_dir = Path(timestamps_dir).name
    videos_dir = f"./output/{output_dir}/videos"


    try:
        os.mkdir(videos_dir)

        titles = tokenize_titles(timestamps_dir)
        trim_vods(titles, vod_dir, videos_dir)
    except:
        print("Videos dir already exists; skipping.")
            
    gen_thumbnail_for_vod(timestamps_dir, "timestamps", r"/home/impasse/Pictures/Altro/Smash/Portraits")