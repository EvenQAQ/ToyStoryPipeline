#coding=utf-8
import math

bed1 = 29322
bal_in_bed1 = 5184
bed2 = 28236
bed3 = 21750
bath = 3843
liv = 13398 + 15555
total_bed = bed1 + bal_in_bed1 + bed2 + bed3
total_pub = bath + liv
total = total_bed + total_pub
fee = 9000

if __name__ == "__main__":
    pub = fee * total_pub / total / 3
    print("pub = ")
    print(pub)

    bed1 = fee * total_bed / total * (bed1 + bal_in_bed1) / total_bed
    print("bed1 = ")
    print(bed1)

    bed2 = fee * total_bed / total * (bed2) / total_bed
    print ("bed2 = ")
    print(bed2)

    bed3 = fee * total_bed / total * (bed3) / total_bed
    print("bed3 = ")
    print(bed3)

    total_fee_calc = pub * 3 + bed1 + bed2 + bed3
    print("total fee = ")
    print (total_fee_calc)
