def rgb_to_hex(rgb_tuple, range_0_1=True, upper=False):
    r, g, b = rgb_tuple
    if range_0_1:
        r = hex(round(r*255))
        g = hex(round(g*255))
        b = hex(round(b*255))
    else:
        r = hex(r)
        g = hex(g)
        b = hex(b)
    hex_ = "".join([x[-2:] if len(x)==4 else "0"+x[-1] for x in [r,g,b]])
    if upper:
        return hex_.upper()
    else:
        return hex_

def calc_color(mask_value, max_mask, min_mask, max_color, min_color):
    m = (min_color - max_color)/(max_mask - min_mask)
    b = min_color - m * max_mask
    color = mask_value * m  + b
    return color


def generate_label(rank: int, element: str):
    u_dict = {"0": "\u2070", "1": "\u00b9",
              "2": "\u00b2", "3": "\u00b3",
              "4": "\u2074", "5": "\u2075",
              "6": "\u2076", "7": "\u2077",
              "8": "\u2078", "9": "\u2079"}
    label = element
    rank_str = str(rank)
    for char in rank_str:
        label += u_dict[char]
    return label

    