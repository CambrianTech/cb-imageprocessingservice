from enum import IntEnum
import cambrian.image_processing as ip

#https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
class SegmentationLabel(IntEnum):
    WALL=0
    BUILDING=1
    SKY=2
    FLOOR=3
    TREE=4
    CEILING=5
    ROAD=6
    BED=7
    WINDOW=8
    GRASS=9
    CABINET=10
    PAVEMENT=11
    PERSON=12
    GROUND=13
    DOOR=14
    TABLE=15
    MOUNTAIN=16
    PLANT=17
    CURTAIN=18
    CHAIR=19
    CAR=20
    WATER=21
    PAINTING=22
    SOFA=23
    SHELF=24
    HOUSE=25
    SEA=26
    MIRROR=27
    RUG=28
    FIELD=29
    ARMCHAIR=30
    SEAT=31
    FENCE=32
    DESK=33
    ROCK=34
    CLOSET=35
    LAMP=36
    BATHTUB=37
    RAILING=38
    CUSHION=39
    PEDASTAL=40
    BOX=41
    COLUMN=42
    SIGN=43
    DRESSER=44
    COUNTER=45
    SAND=46
    SINK=47
    SKYSCRAPER=48
    FIREPLACE=49
    REFRIGERATOR=50
    GRANDSTAND=51
    PATH=52
    STEPS=53
    RUNWAY=54
    SHOWCASE=55
    POOLTABLE=56
    PILLOW=57
    SCREENDOOR=58
    STAIRWAY=59
    RIVER=60
    BRIDGE=61
    BOOKCASE=62
    BLIND=63
    COFFEETABLE=64
    TOILET=65
    FLOWER=66
    BOOK=67
    HILL=68
    BENCH=69
    COUNTERTOP=70
    STOVE=71
    PALMTREE=72
    KITCHENISLAND=73
    COMPUTER=74
    SWIVELCHAIR=75
    BOAT=76
    BAR=77
    ARCADE=78
    SHACK=79
    BUS=80
    TOWEL=81
    LIGHT=82
    TRUCK=83
    TOWER=84
    CHANDELIER=85
    AWNING=86
    STREETLIGHT=87
    BOOTH=88
    TELEVISION=89
    AIRPLANE=90
    DIRTTRACK=91
    CLOTHING=92
    POLE=93
    LAND=94
    BANNISTER=95
    ESCALATOR=96
    OTTOMAN=97
    BOTTLE=98
    BUFFET=99
    POSTER=100
    STAGE=101
    VAN=102
    SHIP=103
    FOUNTAIN=104
    CONVEYOR=105
    CANOPY=106
    WASHINGMACHINE=107
    TOY=108
    POOL=109
    STOOL=110
    BARREL=111
    BASKET=112
    WATERFALL=113
    TENT=114
    BAG=115
    MOTORBIKE=116
    CRADLE=117
    OVEN=118
    BALL=119
    FOOD=120
    STEP=121
    STORAGE=122
    BRANDING=123
    MICROWAVE=124
    POT=125
    ANIMAL=126
    BICYCLE=127
    LAKE=128
    DISHWASHER=129
    SCREEN=130
    BLANKET=131
    SCULPTURE=132
    EXHAUST=133
    SCONCE=134
    VASE=135
    TRAFFICLIGHT=136
    TRAY=137
    TRASH=138
    FAN=139
    PIER=140
    CRT=141
    PLATE=142
    MONITOR=143
    BULLETIN=144
    SHOWER=145
    RADIATOR=146
    GLASS=147
    CLOCK=148
    FLAG=149

class SegmentationSet:
    WALL=[SegmentationLabel.WALL, SegmentationLabel.COLUMN]
    FLOOR=[SegmentationLabel.FLOOR, SegmentationLabel.GROUND, SegmentationLabel.ROAD, SegmentationLabel.PAVEMENT, SegmentationLabel.GRASS, SegmentationLabel.LAND, SegmentationLabel.STAGE]
    CEILING=[SegmentationLabel.CEILING]
    DOOR=[SegmentationLabel.DOOR, SegmentationLabel.SCREENDOOR, SegmentationLabel.CURTAIN]
    TABLE=[SegmentationLabel.TABLE, SegmentationLabel.COFFEETABLE, SegmentationLabel.STEP]

    @classmethod
    def label(cls, _set):
        return _set[0]

    @classmethod
    def name(cls, _set):
        return _set[0].name

    @classmethod
    def value(cls, _set):
        return _set[0].value

    @classmethod
    def color(cls, _set):
        return ip.get_label_color(_set[0])


