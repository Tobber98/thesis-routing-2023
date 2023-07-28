import json
import logging
import logging.config

def main():
    logging.info("Calling main function!")
    logging.debug("You would love it.")
    logging.warning('This is a warning message')
    logging.error('This is an error message')
    logging.critical('This is a critical message')


if __name__ == "__main__":
    with open("logging/settings.json") as fd:
        logging.config.dictConfig(json.load(fd))
    # logging.basicConfig(
    #     filename="test.log", 
    #     level=logging.DEBUG, 
    #     format="[{asctime}][{levelname:^8}] - {message:<50} ({filename}:{lineno:})",
    #     style="{",
    #     datefmt='%Y-%m-%d %H:%M:%S')
        # format="[%(asctime)s] [%(levelname)-8s] --- %(message)s (%(filename)s:%(lineno)s)") # name gives root
    main()




#     random &   \makecell{0.00\% \\(     0)} & \makecell{100.00\% \\(634500)} &   \makecell{0.00\% \\(     0)} \\
#    longest &   \makecell{7.37\% \\( 46743)} &  \makecell{38.53\% \\(244451)} &  \makecell{54.11\% \\(343306)} \\
#         lr &   \makecell{7.45\% \\( 47271)} &  \makecell{38.61\% \\(245003)} &  \makecell{53.94\% \\(342226)} \\
#      min\_x &  \makecell{18.16\% \\(115198)} &  \makecell{43.20\% \\(274097)} &  \makecell{38.65\% \\(245205)} \\
#      min\_y &  \makecell{20.30\% \\(128806)} &  \makecell{43.74\% \\(277511)} &  \makecell{35.96\% \\(228183)} \\
#      min\_z &  \makecell{20.37\% \\(129249)} &  \makecell{43.84\% \\(278139)} &  \makecell{35.79\% \\(227112)} \\
#   min\_axes &  \makecell{26.57\% \\(168601)} &  \makecell{44.17\% \\(280277)} &  \makecell{29.25\% \\(185622)} \\
#     active &  \makecell{32.98\% \\(209258)} &  \makecell{45.72\% \\(290124)} &  \makecell{21.30\% \\(135118)} \\
#   shortest &  \makecell{51.24\% \\(325111)} &  \makecell{43.15\% \\(273782)} &   \makecell{5.61\% \\( 35607)} \\


#     random  &   \makecell{0.00\% \\(     0)} & \makecell{100.00\% \\(634500)} &   \makecell{0.00\% \\(     0)} \\
#         lr  &  \makecell{11.07\% \\( 70263)} &  \makecell{24.56\% \\(155859)} &  \makecell{64.36\% \\(408378)} \\
#    longest  &  \makecell{11.96\% \\( 75891)} &  \makecell{24.98\% \\(158516)} &  \makecell{63.06\% \\(400093)} \\
#      min\_z &  \makecell{37.63\% \\(238764)} &  \makecell{29.85\% \\(189416)} &  \makecell{32.52\% \\(206320)} \\
#      min\_x &  \makecell{39.32\% \\(249462)} &  \makecell{29.93\% \\(189922)} &  \makecell{30.75\% \\(195116)} \\
#      min\_y &  \makecell{40.41\% \\(256430)} &  \makecell{29.94\% \\(189977)} &  \makecell{29.64\% \\(188093)} \\
#   min\_axes &  \makecell{49.34\% \\(313079)} &  \makecell{29.33\% \\(186094)} &  \makecell{21.33\% \\(135327)} \\
#     active  &  \makecell{52.26\% \\(331582)} &  \makecell{29.39\% \\(186481)} &  \makecell{18.35\% \\(116437)} \\
#   shortest  &  \makecell{61.06\% \\(387421)} &  \makecell{27.74\% \\(176012)} &  \makecell{11.20\% \\( 71067)} \\