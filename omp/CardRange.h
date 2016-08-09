#ifndef OMP_CARDRANGE_H
#define OMP_CARDRANGE_H

#include <string>
#include <vector>
#include <array>
#include <cstdint>

namespace omp {

// Stores a set of unique starting hands for Texas Holdem.
class CardRange
{
public:
    // Constructs an empty range.
    CardRange();

    // Constructs a range from an expression. Supported syntax:
    // K4 : all suited and offsuited combos with specified ranks
    // K4s : suited combos
    // K4o : offsuited combos
    // Kc4d : specific suits
    // K4o+ : specified hand and all similar hands with a better kicker (K4 to KQ)
    // 44+ : pocket pair and all higher pairs
    // K4+,Q8s,84 : multiple hands can be combined with comma
    // random : all hands
    // Spaces and non-matching characters in the end are ignored. The expressions are case-insensitive.
    CardRange(const std::string& text);
    CardRange(const char* text);

    // Constructs a range from a list of two-card combinations.
    CardRange(const std::vector<std::array<uint8_t,2>>& combos);

    // Returns a list of card combinations belonging to this range. Guarantees that there are no duplicates.
    // Cards in each combo are ordered so that the bigger rank is always first. The whole vector is sorted in the
    // following order: 1) rank of first card 2) rank of second card 3) suit of first card 4) suit of second card
    const std::vector<std::array<uint8_t,2>>& combinations() const
    {
        return mCombinations;
    }

    // Returns a 64-bit bitmask of cards from a string like "2c8hAh".
    static uint64_t getCardMask(const std::string& text);

private:
    bool parseHand(const char*&p);
    bool parseRank(const char*&p, unsigned& rank);
    bool parseSuit(const char*&p, unsigned& suit);
    bool parseChar(const char*&p, char c);
    void addAll();
    void addCombos(unsigned rank1, unsigned rank2, bool suited, bool offsuited);
    void addCombosPlus(unsigned rank1, unsigned rank2, bool suited, bool offsuited);
    void addCombo(unsigned c1, unsigned c2);
    void removeDuplicates();
    static unsigned charToRank(char c);
    static unsigned charToSuit(char c);

    std::vector<std::array<uint8_t,2>> mCombinations;
};

}

#endif // OMP_CARDRANGE_H
