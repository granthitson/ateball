# IMAGES #

# UI NAVIGATION #
img_back_arrow = "navigation/back_arrow.png"

img_calibration = "misc/calibration.png"

img_close_arrow = "navigation/close_arrow.png"
img_close_red = "navigation/close_red.png"
img_clubs = "misc/clubs.png"

img_confirm = "navigation/confirm.png"
# img_collectCoins = "collectcoins.png"
# img_collectCoins1 = "collectcoins_1.png"

img_friend_challenge = "gamemodes/friends/friend_challenge.png"
img_friend_menu_arrow_left = "navigation/friend_menu_arrow_left.png"
img_friend_menu_arrow_right = "navigation/friend_menu_arrow_right.png"
img_friend_search_icon = "gamemodes/friends/friend_search_icon.png"

img_location_menu_arrow_left = "navigation/location_menu_arrow_left.png"
img_location_menu_arrow_right = "navigation/location_menu_arrow_right.png"

img_lucky_shot = "gamemodes/lucky_shot/lucky_shot.png"
img_lucky_shot_cue_ball = "gamemodes/lucky_shot/lucky_shot_cue_ball.png"
img_lucky_shot_gold_ball = "gamemodes/lucky_shot/lucky_shot_gold_ball.png"
img_lucky_shot_red_ball = "gamemodes/lucky_shot/lucky_shot_red_ball.png"
img_lucky_shot_target = "gamemodes/lucky_shot/lucky_shot_target.png"

img_menu_arrow_left = "navigation/menu_arrow_left.png"
img_menu_arrow_right = "navigation/menu_arrow_right.png"

img_play_guest_btn = "gamemodes/play-guest-btn.png"
img_play_free_lucky_shot_btn = "gamemodes/lucky_shot/play-free_lucky_shot-btn.png"

img_play_special = "gamemodes/play-special-btn.png"

img_settings_btn = "misc/settings-btn.png"
img_gamemenu = "misc/game_menu.png"

img_spin_win = "gamemodes/spin_and_win/spin_win.png"
# img_spinWinSpin = "spinwin_spin.png"
# img_spinWinSpin1 = "spinwin_spin1.png"
# img_spinWinCollect = "spinwin_collect.png"
# img_spinWinIcon = "spinwinicon.png"

img_turn_mask = "game/turn_mask.png"
# UI NAVIGATION #

gamemodes = {
    "CHALLENGE" : {
        "img" : "gamemodes/play-friends-btn.png",
        "index" : 4
    },
    "GUEST" : {
        "img" : "gamemodes/play-guest-btn.png",
        "index" : 1
    },
    "LUCKY_SHOT" : {
        "img" : "gamemodes/play-lucky_shot-btn.png",
        "index" : 3,
        "parent" : "MINIGAMES"
    },
    "MINIGAMES" : {
        "img" : "gamemodes/play-minigames-btn.png",
        "index" : 3
    },
    "NINE_BALL" : {
        "img" : "gamemodes/play-nine_ball-btn.png",
        "index" : 2,
        "start" : "miami",
        "start_index" : 0
    },
    "NO_GUIDELINE" : {
        "img" : "gamemodes/play-no_guideline-btn.png",
        "index" : 1,
        "parent" : "SPECIAL",
        "start" : "beijing",
        "start_index" : 0
    },
    "ONE_ON_ONE" : {
        "img" : "gamemodes/play-1_on_1-btn.png",
        "index" : 0,
        "start" : "las_vegas",
        "start_index" : 5
    },
    "PASS_N_PLAY" : {
        "img" : "gamemodes/play-pass_n_play-btn.png",
        "index" : 5,
        "parent" : "PRACTICE"
    },
    "PRACTICE" : {
        "img" : "gamemodes/play-practice-btn.png",
        "index" : 5
    },
    "QUICK_FIRE" : {
        "img" : "gamemodes/play-quick_fire-btn.png",
        "index" : 5,
        "parent" : "PRACTICE"
    },
    "SPECIAL" : {
        "img" : "gamemodes/play-special-btn.png",
        "index" : 1
    },
    "SPIN_AND_WIN" : {
        "img" : "gamemodes/play-spin_win-btn.png",
        "index" : 3,
        "parent" : "MINIGAMES"
    },
    "TOURNAMENT" : {
        "img" : "gamemodes/play-tournament-btn.png",
        "index" : 1,
        "parent" : "SPECIAL",
        "start" : "buenos_aires",
        "start_index" : 1
    }
}

locations = {
    "amsterdam" : {
        "img" : "locations/location_amsterdam.png",
        "bet" : 5000
    },
    "bangkok" : {
        "img" : "locations/location_bangkok.png",
        "bet" : 5000000
    },
    "barcelona" : {
        "img" : "locations/location_barcelona.png",
        "bet" : 200
    },
    "beijing" : {
        "img" : "locations/location_beijing.png",
        "bet" : "USER-CHOSEN"
    },
    "berlin" : {
        "img" : "locations/location_berlin.png",
        "bet" : 25000000
    },
    "buenos_aires" : {
        "img" : "locations/location_buenos_aires.png",
        "bet" : 500
    },
    "cairo" : {
        "img" : "locations/location_cairo.png",
        "bet" : 250000
    },
    "dallas" : {
        "img" : "locations/location_dallas.png",
        "bet" : "USER-CHOSEN"
    },
    "dubai" : {
        "img" : "locations/location_dubai.png",
        "bet" : 500000
    },
    "istanbul" : {
        "img" : "locations/location_istanbul.png",
        "bet" : "USER-CHOSEN"
    },
    "jakarta" : {
        "img" : "locations/location_jakarta.png",
        "bet" : 50000
    },
    "las_vegas" : {
        "img" : "locations/location_las_vegas.png",
        "bet" : 10000
    },
    "london" : {
        "img" : "locations/location_london.png",
        "bet" : 50
    },
    "miami_beach" : {
        "img" : "locations/location_miami_beach.png",
        "bet" : "USER-CHOSEN"
    },
    "monaco" : {
        "img" : "locations/location_monaco.png",
        "bet" : "ALL-IN",
        "min" : 1000
    },
    "moscow" : {
        "img" : "locations/location_moscow.png",
        "bet" : 500
    },
    "mumbai" : {
        "img" : "locations/location_mumbai.png",
        "bet" : 15000000
    },
    "paris" : {
        "img" : "locations/location_paris.png",
        "bet" : 2500000
    },
    "rio" : {
        "img" : "locations/location_rio.png",
        "bet" : 3000
    },
    "rome" : {
        "img" : "locations/location_rome.png",
        "bet" : 4000000
    },
    "seoul" : {
        "img" : "locations/location_seoul.png",
        "bet" : 10000000
    },
    "shanghai" : {
        "img" : "locations/location_shanghai.png",
        "bet" : 1000000
    },
    "singapore" : {
        "img" : "locations/location_singapore.png",
        "bet" : 10000
    },
    "sydney" : {
        "img" : "locations/location_sydney.png",
        "bet" : 100
    },
    "tokyo" : {
        "img" : "locations/location_tokyo.png",
        "bet" : 2500
    },
    "toronto" : {
        "img" : "locations/location_toronto.png",
        "bet" : 100000
    }
}

bets = {
    100 : {
        "img" : "bets/100.png"
    },
    500 : {
        "img" : "bets/500.png"
    },
    2500 : {
        "img" : "bets/2500.png"
    },
    10000 : {
        "img" : "bets/10000.png"
    },
    50000 : {
        "img" : "bets/50000.png"
    },
    100000 : {
        "img" : "bets/100000.png"
    },
    500000 : {
        "img" : "bets/500000.png"
    },
    2500000 : {
        "img" : "bets/2500000.png"
    },
    10000000 : {
        "img" : "bets/10000000.png"
    }
}

# GAME NAVIGATION #
img_1ball = "1ball.png"
img_2ball = "2ball.png"
img_3ball = "3ball.png"
img_4ball = "4ball.png"
img_5ball = "5ball.png"
img_6ball = "6ball.png"
img_7ball = "7ball.png"
img_8ball = "8ball.png"
img_9ball = "9ball.png"
img_10ball = "10ball.png"
img_11ball = "11ball.png"
img_12ball = "12ball.png"
img_13ball = "13ball.png"
img_14ball = "14ball.png"
img_15ball = "15ball.png"
img_1ballDark = "1ballDark.png"
img_2ballDark = "2ballDark.png"
img_3ballDark = "3ballDark.png"
img_4ballDark = "4ballDark.png"
img_5ballDark = "5ballDark.png"
img_6ballDark = "6ballDark.png"
img_7ballDark = "7ballDark.png"
img_8ballDark = "8ballDark.png"
img_9ballDark = "9ballDark.png"
img_10ballDark = "10ballDark.png"
img_11ballDark = "11ballDark.png"
img_12ballDark = "12ballDark.png"
img_13ballDark = "13ballDark.png"
img_14ballDark = "14ballDark.png"
img_15ballDark = "15ballDark.png"
img_cueball = "cueball.png"
img_eightball = "eightball.png"

img_ballPic1 = "ballpic1.png"

img_tlh = "tlh.png"
img_tmh = "tmh.png"
img_trh = "trh.png"
img_blh = "blh.png"
img_bmh = "bmh.png"
img_brh = "brh.png"

img_luckyShotTRH = "luckyTRH.png"
img_luckyShotTMH = "luckyTMH.png"
img_luckyShotTLH = "luckyTLH.png"
img_luckyShotBLH = "luckyBLH.png"
img_luckyShotBMH = "luckyBMH.png"
img_luckyShotBRH = "luckyBRH.png"

img_topRail = "toprail.png"
img_bottomRail = "bottomrail.png"
img_leftRail = "leftrail.png"
img_rightRail = "rightrail.png"
# GAME NAVIGATION #

# IMAGES #

# GAME #
game_width = 900
game_height = 600

# offsetx, offsety, width, height
table_dims = (106, 176, 690, 360)

round_time = 30

# offsetx, offsety, width, height
pocketed_dims = (725, 0, 50, table_dims[3])

# offsetx, offsety, width, height
bot_targets_dims = (7, 119, 210, 30)
opponent_targets_offset = (465, 0)

hole_locations = [ 
    ["trh", img_trh, [666, 24]],
    ["tmh", img_tmh, [345, 4]],
    ["tlh", img_tlh, [23, 24]],
    ["blh", img_blh, [23, 336]],
    ["bmh", img_bmh, [345, 356]],
    ["brh", img_brh, [666, 336]] 
]

ball_diameter = 12 #roughly 11/12

# BALLS #

img_glove = "glove.png"
img_dontHit = "dontHit.png"

solids = {
    img_1ball: "yellow",
    img_2ball: "blue",
    img_3ball: "lightred", 
    img_4ball: "purple", 
    img_5ball: "orange", 
    img_6ball: "green", 
    img_7ball: "darkred"
}

solidsDark = {
    img_1ballDark: "yellow", 
    img_2ballDark: "blue", 
    img_3ballDark: "lightred", 
    img_4ballDark: "purple", 
    img_5ballDark: "orange", 
    img_6ballDark: "green",
    img_7ballDark: "darkred"
}

stripes = {
    img_9ball: "yellow", 
    img_10ball: "blue", 
    img_11ball: "lightred", 
    img_12ball: "purple", 
    img_13ball: "orange", 
    img_14ball: "green", 
    img_15ball: "darkred"
}

stripesDark = {
    img_9ballDark: "yellow", 
    img_10ballDark: "blue", 
    img_11ballDark: "lightred", 
    img_12ballDark: "purple", 
    img_13ballDark: "orange", 
    img_14ballDark: "green", 
    img_15ballDark: "darkred"
}

# BALLS #

# GAME #

debug = False