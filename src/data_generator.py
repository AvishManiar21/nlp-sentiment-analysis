"""
Synthetic tech product review generator.
Produces 50,000+ realistic reviews across multiple product categories
with varied vocabulary, ratings, and sentiment patterns.
"""

import random
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

CATEGORIES = {
    "Smartphones": {
        "brands": ["Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Sony", "Motorola"],
        "products": [
            "iPhone 16 Pro", "Galaxy S25 Ultra", "Pixel 9 Pro", "OnePlus 13",
            "Xiaomi 15", "Xperia 1 VI", "Edge 50 Ultra", "iPhone 16",
            "Galaxy S25", "Pixel 9", "OnePlus 13R", "Xiaomi 15 Pro"
        ],
        "aspects": ["battery life", "camera quality", "display", "performance", "build quality",
                     "software", "charging speed", "fingerprint sensor", "face unlock", "speaker quality"],
    },
    "Laptops": {
        "brands": ["Apple", "Dell", "Lenovo", "HP", "ASUS", "Acer", "Microsoft"],
        "products": [
            "MacBook Pro 16", "MacBook Air M3", "XPS 15", "ThinkPad X1 Carbon",
            "Spectre x360", "ROG Zephyrus", "Swift Go 16", "Surface Laptop 6",
            "Dell Inspiron 16", "Lenovo Yoga 9i", "HP Envy 16", "ASUS Zenbook 14"
        ],
        "aspects": ["keyboard", "trackpad", "display quality", "battery life", "performance",
                     "build quality", "fan noise", "port selection", "weight", "webcam"],
    },
    "Headphones": {
        "brands": ["Sony", "Bose", "Apple", "Sennheiser", "JBL", "Jabra", "Samsung"],
        "products": [
            "WH-1000XM6", "QuietComfort Ultra", "AirPods Max", "Momentum 4",
            "Tour One M2", "Elite 85t", "Galaxy Buds3 Pro", "WF-1000XM6",
            "AirPods Pro 3", "Bose Ultra Open", "JBL Live 770NC", "Jabra Elite 10"
        ],
        "aspects": ["sound quality", "noise cancellation", "comfort", "battery life",
                     "call quality", "connectivity", "fit", "bass response", "app features", "case design"],
    },
    "Smartwatches": {
        "brands": ["Apple", "Samsung", "Garmin", "Fitbit", "Google", "Amazfit", "Huawei"],
        "products": [
            "Apple Watch Ultra 3", "Galaxy Watch 7", "Garmin Fenix 8", "Fitbit Sense 3",
            "Pixel Watch 3", "Amazfit T-Rex Ultra", "Huawei Watch GT 5",
            "Apple Watch Series 10", "Galaxy Watch FE", "Garmin Venu 4"
        ],
        "aspects": ["fitness tracking", "heart rate accuracy", "battery life", "display",
                     "app ecosystem", "GPS accuracy", "sleep tracking", "water resistance",
                     "band comfort", "notifications"],
    },
    "Tablets": {
        "brands": ["Apple", "Samsung", "Microsoft", "Lenovo", "Amazon", "Google", "Xiaomi"],
        "products": [
            "iPad Pro M4", "iPad Air M2", "Galaxy Tab S10 Ultra", "Surface Pro 10",
            "Tab P12 Pro", "Fire HD 10", "Pixel Tablet", "Xiaomi Pad 7 Pro",
            "iPad Mini 7", "Galaxy Tab S10", "Surface Go 4", "Lenovo Tab Plus"
        ],
        "aspects": ["display quality", "stylus support", "performance", "battery life",
                     "app availability", "multitasking", "speaker quality", "portability",
                     "keyboard support", "value for money"],
    },
}

POSITIVE_TEMPLATES = [
    "Absolutely love the {aspect} on this {product}. {detail}",
    "The {aspect} is outstanding. {detail} Highly recommend!",
    "I'm blown away by the {aspect}. {detail} Best purchase I've made.",
    "The {aspect} exceeds all expectations. {detail}",
    "Can't say enough good things about the {aspect}. {detail} Worth every penny.",
    "After using {product} for {duration}, the {aspect} is still fantastic. {detail}",
    "Upgraded from my old device and the {aspect} improvement is incredible. {detail}",
    "The {product} nails it with its {aspect}. {detail} Five stars!",
    "{product} delivers on its promise. The {aspect} is top-notch. {detail}",
    "Very impressed with the {aspect}. {detail} Would buy again without hesitation.",
    "The {aspect} alone makes this worth buying. {detail}",
    "Coming from a competitor, I'm amazed at the {aspect}. {detail}",
    "Everything about the {aspect} is perfect. {detail} No complaints at all.",
    "Super happy with the {aspect} on my new {product}. {detail}",
    "Great {aspect}! {detail} This {product} is a game changer.",
]

NEGATIVE_TEMPLATES = [
    "Really disappointed with the {aspect}. {detail}",
    "The {aspect} is terrible on this {product}. {detail} Considering returning it.",
    "Don't buy this for the {aspect}. {detail} Total letdown.",
    "The {aspect} is the weakest point of {product}. {detail}",
    "After {duration} of use, the {aspect} has degraded significantly. {detail}",
    "Expected much better {aspect} at this price point. {detail}",
    "The {aspect} is a deal-breaker for me. {detail} Very frustrated.",
    "Regret buying the {product}. The {aspect} is awful. {detail}",
    "Worst {aspect} I've experienced in any device. {detail}",
    "The {product} fails miserably at {aspect}. {detail} Save your money.",
    "Had high hopes but the {aspect} ruined the experience. {detail}",
    "Can't believe how bad the {aspect} is. {detail} Not worth the price.",
    "The {aspect} issues make this {product} unusable for me. {detail}",
    "Returned the {product} because of poor {aspect}. {detail}",
    "Seriously underwhelmed by the {aspect}. {detail} Looking at alternatives now.",
]

NEUTRAL_TEMPLATES = [
    "The {aspect} is decent but nothing special. {detail}",
    "Average {aspect} on the {product}. {detail} It gets the job done.",
    "The {aspect} is okay for the price. {detail}",
    "Mixed feelings about the {aspect}. {detail} It's acceptable.",
    "The {aspect} on {product} is neither great nor terrible. {detail}",
    "Not bad, not amazing. The {aspect} is just adequate. {detail}",
    "The {aspect} meets basic expectations. {detail} Nothing more, nothing less.",
    "For what it costs, the {aspect} is fair. {detail}",
    "The {aspect} is middle-of-the-road. {detail} Could be better, could be worse.",
    "Decent {aspect} overall. {detail} Just don't expect miracles.",
]

POSITIVE_DETAILS = {
    "battery life": [
        "Easily lasts a full day with heavy use.",
        "I only need to charge it every two days.",
        "Battery optimization is excellent.",
        "Fast charging gets me to 80% in under 30 minutes.",
        "Even with GPS and streaming, it holds up all day.",
    ],
    "camera quality": [
        "Photos are incredibly sharp even in low light.",
        "Video stabilization is best-in-class.",
        "The portrait mode produces DSLR-quality bokeh.",
        "Night mode blew me away with the detail it captures.",
        "Colors look natural and true to life.",
    ],
    "display": [
        "The colors are vibrant and incredibly accurate.",
        "Brightness is perfect even in direct sunlight.",
        "120Hz refresh rate makes everything buttery smooth.",
        "HDR content looks absolutely stunning.",
        "The resolution is razor sharp with no visible pixels.",
    ],
    "performance": [
        "Apps open instantly with zero lag.",
        "Multitasking is seamless even with many apps open.",
        "Gaming performance is console-level quality.",
        "No stuttering or frame drops whatsoever.",
        "Handles everything I throw at it effortlessly.",
    ],
    "build quality": [
        "Feels premium and solid in hand.",
        "The materials used are top-quality.",
        "Survived a couple of drops with no damage.",
        "The finish and attention to detail are superb.",
        "Feels like it will last for years.",
    ],
    "sound quality": [
        "Rich, detailed audio with incredible clarity.",
        "Bass is deep and punchy without being muddy.",
        "Spatial audio is a game-changing experience.",
        "Every instrument is clearly separated in the mix.",
        "Best audio I've heard from any consumer product.",
    ],
    "noise cancellation": [
        "Blocks out everything, even airplane engine noise.",
        "ANC is incredibly effective in noisy environments.",
        "Transparency mode is natural-sounding.",
        "Can't hear anything around me when it's turned on.",
        "The adaptive noise cancellation adjusts perfectly.",
    ],
    "comfort": [
        "Can wear them for hours without any discomfort.",
        "The ergonomic design fits perfectly.",
        "So light I forget I'm wearing them.",
        "Padding is plush and never causes pressure points.",
        "Best comfort I've experienced in this category.",
    ],
    "fitness tracking": [
        "Step counting is spot-on compared to manual counting.",
        "Workout detection is accurate and automatic.",
        "The health insights are genuinely useful.",
        "Heart rate monitoring matches clinical-grade devices.",
        "Tracks every metric I care about reliably.",
    ],
    "display quality": [
        "Colors pop and text is crisp at any size.",
        "The anti-glare coating works beautifully.",
        "Viewing angles are excellent from any position.",
        "ProMotion makes scrolling incredibly smooth.",
        "Mini-LED backlighting provides stunning contrast.",
    ],
    "software": [
        "Clean, intuitive interface with no bloatware.",
        "Regular updates keep it running smoothly.",
        "The ecosystem integration is seamless.",
        "Software optimization is top-tier.",
        "Every feature works exactly as advertised.",
    ],
    "keyboard": [
        "Typing experience is best-in-class.",
        "Keys have perfect travel and satisfying feedback.",
        "Backlight is even and adjustable.",
        "Layout is well thought out and comfortable.",
        "Can type for hours without fatigue.",
    ],
    "trackpad": [
        "Gestures are smooth and responsive.",
        "Glass surface feels premium and precise.",
        "Haptic feedback is incredibly realistic.",
        "Best trackpad on any laptop, period.",
        "Palm rejection works flawlessly.",
    ],
    "connectivity": [
        "Bluetooth connection is rock-solid and stable.",
        "Pairs instantly with all my devices.",
        "Multipoint connection switches seamlessly.",
        "Range is excellent even through walls.",
        "Never had a single dropout or disconnection.",
    ],
    "charging speed": [
        "Goes from 0 to 100% in under an hour.",
        "Quick charge gives hours of use in minutes.",
        "Wireless charging is fast and convenient.",
        "Charging speed is the fastest I've seen.",
        "The included charger is excellent quality.",
    ],
    "value for money": [
        "Incredible specs for the price point.",
        "Outperforms devices twice its price.",
        "Best bang for your buck in this category.",
        "You'd have to spend double to get anything better.",
        "Price-to-performance ratio is unbeatable.",
    ],
}

NEGATIVE_DETAILS = {
    "battery life": [
        "Barely makes it through half a day.",
        "Dies before lunch with normal use.",
        "Battery drain is excessive and unpredictable.",
        "Have to carry a charger everywhere I go.",
        "Battery degraded noticeably after just a few months.",
    ],
    "camera quality": [
        "Photos are blurry and lack detail.",
        "Colors look washed out and unnatural.",
        "Night mode is basically useless.",
        "Video quality is poor with lots of noise.",
        "Shutter lag causes me to miss moments.",
    ],
    "display": [
        "Screen is dim and hard to read outdoors.",
        "Colors look off and overly saturated.",
        "Noticeable backlight bleed around the edges.",
        "Refresh rate stutters during normal use.",
        "Resolution feels dated compared to competitors.",
    ],
    "performance": [
        "Constant lag and stuttering throughout the UI.",
        "Apps crash frequently and take forever to load.",
        "Overheats during basic tasks.",
        "RAM management is terrible, apps reload constantly.",
        "Gets noticeably slower with each software update.",
    ],
    "build quality": [
        "Feels cheap and plasticky in hand.",
        "Creaks and flexes with normal handling.",
        "Paint started chipping after a week.",
        "The hinge feels like it'll break any day.",
        "Quality control is clearly lacking.",
    ],
    "sound quality": [
        "Sounds tinny and lacks any bass.",
        "Audio distorts at higher volumes.",
        "Muffled and unclear compared to competitors.",
        "The equalizer options can't fix the poor tuning.",
        "Expected much better audio at this price.",
    ],
    "noise cancellation": [
        "Barely blocks any ambient noise.",
        "ANC introduces an annoying hiss.",
        "Wind noise passes right through.",
        "Nowhere near as effective as advertised.",
        "Transparency mode sounds robotic and unnatural.",
    ],
    "comfort": [
        "Hurts my ears after just 30 minutes.",
        "Way too tight and causes headaches.",
        "The materials make my skin sweaty and irritated.",
        "Keeps falling off during movement.",
        "Pressure points develop quickly during use.",
    ],
    "fitness tracking": [
        "Step count is wildly inaccurate.",
        "Heart rate readings don't match my chest strap.",
        "Sleep tracking is basically random numbers.",
        "Misses half my workouts even with auto-detect on.",
        "Calorie estimates are laughably wrong.",
    ],
    "display quality": [
        "Washed out and lacking contrast.",
        "Terrible viewing angles, colors shift immediately.",
        "Bezels are massive and dated-looking.",
        "Screen flickers at low brightness.",
        "Touch response is laggy and misregisters taps.",
    ],
    "software": [
        "Buggy and crashes constantly.",
        "Loaded with bloatware that can't be removed.",
        "Updates break more things than they fix.",
        "Interface is confusing and unintuitive.",
        "Features promised at launch are still missing.",
    ],
    "keyboard": [
        "Keys feel mushy with no tactile feedback.",
        "Keyboard flex is terrible when typing.",
        "Layout has frustrating non-standard key placement.",
        "Backlight is uneven with light bleed.",
        "Keys started double-typing after a month.",
    ],
    "trackpad": [
        "Rattles and clicks unevenly.",
        "Palm rejection is non-existent.",
        "Gestures are unreliable and laggy.",
        "Surface feels rough and cheap.",
        "Cursor jumping makes precise work impossible.",
    ],
    "connectivity": [
        "Bluetooth drops connection every few minutes.",
        "Pairing is a frustrating experience every time.",
        "Can't maintain connection beyond a few feet.",
        "Multipoint switching doesn't work reliably.",
        "Constant audio stuttering and cutouts.",
    ],
    "charging speed": [
        "Takes forever to charge fully.",
        "Fast charging claims are completely exaggerated.",
        "Wireless charging barely works.",
        "Charger gets dangerously hot.",
        "Charging port is already becoming loose.",
    ],
    "value for money": [
        "Way overpriced for what you get.",
        "Better options available for half the price.",
        "Feels like you're paying for the brand name only.",
        "Not worth the premium price tag.",
        "The budget version from another brand outperforms this.",
    ],
}

DURATIONS = ["a week", "two weeks", "a month", "three months", "six months", "a year"]


def _get_detail(aspect, sentiment):
    if sentiment == "positive":
        pool = POSITIVE_DETAILS
    else:
        pool = NEGATIVE_DETAILS

    generic_positive = [
        "Really pleased overall.", "Exceeded my expectations.",
        "Works as advertised.", "Very satisfied with this aspect.",
        "No issues whatsoever.", "Couldn't be happier.",
    ]
    generic_negative = [
        "Very disappointing.", "Not what I expected at all.",
        "Needs serious improvement.", "Below average in every way.",
        "Would not recommend based on this.", "Frustrating experience.",
    ]

    if aspect in pool:
        return random.choice(pool[aspect])
    return random.choice(generic_positive if sentiment == "positive" else generic_negative)


def generate_review(category_name, category_data):
    product = random.choice(category_data["products"])
    brand = next((b for b in category_data["brands"] if any(
        product.startswith(prefix) for prefix in [b, b.split()[0]]
    )), random.choice(category_data["brands"]))

    sentiment_roll = random.random()
    if sentiment_roll < 0.42:
        sentiment = "positive"
        rating = random.choices([4, 5], weights=[35, 65])[0]
        template = random.choice(POSITIVE_TEMPLATES)
    elif sentiment_roll < 0.72:
        sentiment = "negative"
        rating = random.choices([1, 2], weights=[55, 45])[0]
        template = random.choice(NEGATIVE_TEMPLATES)
    else:
        sentiment = "neutral"
        rating = random.choice([3, 3, 3, 4, 2])
        template = random.choice(NEUTRAL_TEMPLATES)

    aspect = random.choice(category_data["aspects"])
    detail = _get_detail(aspect, "positive" if sentiment == "positive" else "negative")
    duration = random.choice(DURATIONS)

    review_text = template.format(
        aspect=aspect, product=product, detail=detail, duration=duration
    )

    if random.random() < 0.35:
        second_aspect = random.choice([a for a in category_data["aspects"] if a != aspect])
        if sentiment == "positive":
            review_text += f" Also, the {second_aspect} is great."
        elif sentiment == "negative":
            review_text += f" The {second_aspect} is also lacking."
        else:
            review_text += f" The {second_aspect} is alright too."

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 2, 28)
    review_date = start_date + timedelta(
        days=random.randint(0, (end_date - start_date).days)
    )

    verified = random.random() < 0.78
    helpful_votes = int(random.expovariate(0.3)) if random.random() < 0.5 else 0

    return {
        "review_id": None,
        "product": product,
        "brand": brand,
        "category": category_name,
        "rating": rating,
        "review_text": review_text,
        "review_date": review_date.strftime("%Y-%m-%d"),
        "verified_purchase": verified,
        "helpful_votes": helpful_votes,
    }


def generate_dataset(num_reviews=52000, output_path=None):
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "tech_reviews.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    category_weights = {
        "Smartphones": 0.30,
        "Laptops": 0.25,
        "Headphones": 0.20,
        "Smartwatches": 0.13,
        "Tablets": 0.12,
    }

    reviews = []
    for cat_name, weight in category_weights.items():
        cat_count = int(num_reviews * weight)
        cat_data = CATEGORIES[cat_name]
        for _ in range(cat_count):
            reviews.append(generate_review(cat_name, cat_data))

    while len(reviews) < num_reviews:
        cat_name = random.choice(list(CATEGORIES.keys()))
        reviews.append(generate_review(cat_name, CATEGORIES[cat_name]))

    random.shuffle(reviews)
    for idx, review in enumerate(reviews, 1):
        review["review_id"] = f"REV-{idx:06d}"

    fieldnames = [
        "review_id", "product", "brand", "category", "rating",
        "review_text", "review_date", "verified_purchase", "helpful_votes"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews)

    print(f"Generated {len(reviews):,} reviews -> {output_path}")
    return output_path


if __name__ == "__main__":
    generate_dataset()
