#ifndef OMP_CONSTANTS_H
#define OMP_CONSTANTS_H

namespace omp {

static const unsigned MAX_PLAYERS = 6;

static const unsigned CARD_COUNT = 52;
static const unsigned RANK_COUNT = 13;
static const unsigned SUIT_COUNT = 4;
static const unsigned SUIT_MASK = 0x3;
static const unsigned RANK_MASK = 0x3c;
static const unsigned RANK_SHIFT = 2;
static const unsigned BOARD_CARDS = 5;
static const unsigned COMBO_COUNT = 1326;

static const unsigned HAND_CATEGORY_OFFSET = 0x1000; // 4096
static const unsigned HAND_CATEGORY_SHIFT = 12;
static const unsigned HIGH_CARD = 1 * HAND_CATEGORY_OFFSET;
static const unsigned PAIR = 2 * HAND_CATEGORY_OFFSET;
static const unsigned TWO_PAIR = 3 * HAND_CATEGORY_OFFSET;
static const unsigned THREE_OF_A_KIND = 4 * HAND_CATEGORY_OFFSET;
static const unsigned STRAIGHT = 5 * HAND_CATEGORY_OFFSET;
static const unsigned FLUSH = 6 * HAND_CATEGORY_OFFSET;
static const unsigned FULL_HOUSE = 7 * HAND_CATEGORY_OFFSET;
static const unsigned FOUR_OF_A_KIND = 8 * HAND_CATEGORY_OFFSET;
static const unsigned STRAIGHT_FLUSH = 9 * HAND_CATEGORY_OFFSET;

}

#endif // OMP_CONSTANTS_H
