id_1 = "308106707"
id_2 = "200397354"
id_3 = "303016661"
email = "dvirginz@gmail.com"

def get_details():
    if (not id_1) or (not id_2) or not (email):
        raise Exception("Missing submitters info")

    info = str.format("{}_{}      email: {}", id_1, id_2, email)

    return info