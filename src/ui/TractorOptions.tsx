// The match options form, shared by the local and online lobbies (online
// guests see it read-only inside a disabled fieldset).

import { useState, type ReactNode } from 'react';
import type { Seat } from '@/core/types';
import { DECK_SIZE, LEVELS, RANKS, levelLabel, type Level, type Rank } from '@/game/cards';
import { MAX_DECKS, resolveRules, validateConfig, type TractorConfig } from '@/game/config';

interface Props {
  config: TractorConfig;
  seats: readonly Seat[];
  onChange: (config: TractorConfig) => void;
  readOnly?: boolean;
}

function Row({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="opt-row">
      <span className="opt-label">
        {label}
        {hint && <span className="opt-hint">{hint}</span>}
      </span>
      <span className="opt-control">{children}</span>
    </label>
  );
}

function Select<T extends string | number>({ value, options, onChange }: {
  value: T;
  options: ReadonlyArray<readonly [T, string]>;
  onChange: (v: T) => void;
}) {
  return (
    <select value={String(value)} onChange={(e) => onChange(options.find(([v]) => String(v) === e.target.value)![0])}>
      {options.map(([v, label]) => (
        <option key={String(v)} value={String(v)}>{label}</option>
      ))}
    </select>
  );
}

export function TractorOptions({ config, seats, onChange }: Props) {
  const [advanced, setAdvanced] = useState(false);
  const n = seats.length;
  const set = <K extends keyof TractorConfig>(k: K, v: TractorConfig[K]) => onChange({ ...config, [k]: v });
  const errors = validateConfig(config, seats);
  let summary = '';
  if (errors.length === 0) {
    const r = resolveRules(config, seats);
    const perHand = (r.numDecks * DECK_SIZE - r.cardsToRemove - r.kittySize) / n;
    summary = `${r.numDecks} deck${r.numDecks === 1 ? '' : 's'} · ${perHand} cards each · kitty of ${r.kittySize}`
      + (r.cardsToRemove ? ` · ${r.cardsToRemove} low cards removed` : '')
      + (r.gameMode === 'findingFriends' ? ` · landlord calls ${r.numFriends} friend${r.numFriends === 1 ? '' : 's'}` : ' · fixed partners (opposite seats)')
      + ` · ${r.biddingStyle === 'live' ? 'bid while dealing' : 'bid in turns'}`;
  }
  const deckOptions: Array<readonly [number, string]> = [[0, `Auto (${Math.max(1, Math.floor(n / 2))})`]];
  for (let d = 1; d <= Math.min(MAX_DECKS, Math.max(1, n * 2)); d++) deckOptions.push([d, String(d)]);
  const friendOptions: Array<readonly [number, string]> = [[-1, `Auto (${Math.max(0, Math.floor(n / 2) - 1)})`]];
  for (let f = 0; f <= Math.max(0, n - 2); f++) friendOptions.push([f, String(f)]);

  return (
    <div className="tractor-options">
      <Row label="Game" hint={n % 2 === 1 ? 'Odd tables play Finding Friends' : undefined}>
        <Select
          value={config.gameMode}
          options={[
            ['tractor', 'Tractor (fixed partners)'],
            ['findingFriends', 'Finding Friends (landlord calls partners)'],
          ]}
          onChange={(v) => set('gameMode', v)}
        />
      </Row>
      {config.gameMode === 'findingFriends' && (
        <Row label="Friends called">
          <Select value={config.numFriends ?? -1} options={friendOptions} onChange={(v) => set('numFriends', v < 0 ? null : v)} />
        </Row>
      )}
      <Row label="Decks">
        <Select value={config.numDecks ?? 0} options={deckOptions} onChange={(v) => set('numDecks', v === 0 ? null : v)} />
      </Row>
      <Row label="Kitty size" hint="Auto: leftover cards (at least 5). Other sizes remove low cards.">
        <span className="opt-inline">
          <input
            type="checkbox"
            checked={config.kittySize === null}
            onChange={(e) => set('kittySize', e.target.checked ? null : resolveKitty(config, seats))}
            aria-label="Automatic kitty size"
          />
          <span>auto</span>
          {config.kittySize !== null && (
            <input
              type="number"
              min={0}
              max={40}
              value={config.kittySize}
              onChange={(e) => set('kittySize', Math.max(0, Math.floor(Number(e.target.value) || 0)))}
              style={{ width: 70 }}
            />
          )}
        </span>
      </Row>
      <Row label="Declaring trump" hint="Hot-seat tables always bid in turns.">
        <Select
          value={config.biddingStyle}
          options={[
            ['auto', 'Auto'],
            ['live', 'While dealing (live)'],
            ['turns', 'After the deal, in turns'],
          ]}
          onChange={(v) => set('biddingStyle', v)}
        />
      </Row>
      <Row label="Ranks">
        <span className="opt-inline">
          <Select value={config.startingRank} options={RANKS.map((r) => [r, levelLabel(r)] as const)} onChange={(v) => set('startingRank', v as Rank)} />
          <span>to</span>
          <Select value={config.maxRank} options={LEVELS.map((r) => [r, levelLabel(r)] as const)} onChange={(v) => set('maxRank', v as Level)} />
        </span>
      </Row>

      <Row label="Hide the defenders' points" hint="Keep count yourself: their captured points show at the end of the round.">
        <input type="checkbox" checked={config.hideLandlordPoints} onChange={(e) => set('hideLandlordPoints', e.target.checked)} />
      </Row>
      <Row label="Hide played cards" hint="The log says who played how many cards, not which.">
        <input type="checkbox" checked={config.hidePlayedCards} onChange={(e) => set('hidePlayedCards', e.target.checked)} />
      </Row>

      <button type="button" className="secondary tiny opt-advanced-toggle" onClick={() => setAdvanced((v) => !v)}>
        {advanced ? '▾ Fewer options' : '▸ More options (house rules)'}
      </button>
      {advanced && (
        <div className="opt-advanced">
          <Row label="Overcalling a bid">
            <Select
              value={config.bidPolicy}
              options={[
                ['jokerOrGreaterLength', 'More cards, or jokers at equal length'],
                ['jokerOrHigherSuit', '…or a higher suit (♠>♥>♣>♦) at equal length'],
                ['greaterLength', 'Only more cards'],
              ]}
              onChange={(v) => set('bidPolicy', v)}
            />
          </Row>
          <Row label="Joker bids (no trump)">
            <Select
              value={config.jokerBidPolicy}
              options={[
                ['bothTwoOrMore', 'A pair or more'],
                ['bothNumDecks', 'One per deck'],
                ['ljNumDecksHjNumDecksLessOne', 'Small: one per deck, big: one fewer'],
                ['disabled', 'Not allowed'],
              ]}
              onChange={(v) => set('jokerBidPolicy', v)}
            />
          </Row>
          <Row label="Your own winning bid">
            <Select
              value={config.bidReinforcementPolicy}
              options={[
                ['reinforceWhileWinning', 'May only be reinforced'],
                ['overturnOrReinforceWhileWinning', 'May be reinforced or changed'],
                ['reinforceWhileEquivalent', 'May be reinforced to tie an overcall'],
              ]}
              onChange={(v) => set('bidReinforcementPolicy', v)}
            />
          </Row>
          <Row label="First landlord">
            <Select
              value={config.firstLandlordSelection}
              options={[
                ['winningBid', 'The winning bidder'],
                ['firstBid', 'The first bidder'],
              ]}
              onChange={(v) => set('firstLandlordSelection', v)}
            />
          </Row>
          <Row label="Nobody bids" hint="With a landlord, kitty cards are turned over to set trump.">
            <Select
              value={config.kittyBidPolicy}
              options={[
                ['firstCard', 'First kitty card sets trump'],
                ['firstCardOfLevelOrHighest', 'First card of the level (or the highest)'],
              ]}
              onChange={(v) => set('kittyBidPolicy', v)}
            />
          </Row>
          <Row label="Following format">
            <Select
              value={config.trickDrawPolicy}
              options={[
                ['noProtections', 'Match the lead (break tuples if needed)'],
                ['longerTuplesProtected', "Triples needn't be broken for pairs"],
                ['onlyDrawTractorOnTractor', 'Only tractors draw out tractors'],
                ['longerTuplesProtectedAndOnlyDrawTractorOnTractor', 'Both protections'],
                ['noFormatBasedDraw', 'Only follow suit'],
              ]}
              onChange={(v) => set('trickDrawPolicy', v)}
            />
          </Row>
          <Row label="Beating a throw">
            <Select
              value={config.throwEvaluationPolicy}
              options={[
                ['all', 'Every unit must be beaten'],
                ['highest', 'The highest card decides'],
                ['trickUnitLength', 'Longest units decide'],
              ]}
              onChange={(v) => set('throwEvaluationPolicy', v)}
            />
          </Row>
          <Row label="Failed throw">
            <Select
              value={config.throwPenalty}
              options={[
                ['none', 'No penalty'],
                ['tenPointsPerAttempt', '10 points to the other side'],
              ]}
              onChange={(v) => set('throwPenalty', v)}
            />
          </Row>
          <Row label="Kitty multiplier" hint="Points in the kitty, if the attackers win the last trick.">
            <Select
              value={config.kittyPenalty}
              options={[
                ['times', '2 × the last trick’s largest unit'],
                ['power', '2 ^ the last trick’s largest unit'],
              ]}
              onChange={(v) => set('kittyPenalty', v)}
            />
          </Row>
          <Row label="Advancing">
            <Select
              value={config.advancementPolicy}
              options={[
                ['unrestricted', 'Must defend the final rank'],
                ['defendPoints', 'Also defend 5, 10 and K'],
                ['fullyUnrestricted', 'No defending needed'],
              ]}
              onChange={(v) => set('advancementPolicy', v)}
            />
          </Row>
          {config.gameMode === 'findingFriends' && (
            <>
              <Row label="Friend calls">
                <Select
                  value={config.friendSelectionPolicy}
                  options={[
                    ['unrestricted', 'Any non-trump card'],
                    ['trumpsIncluded', 'Any card'],
                    ['highestCardNotAllowed', 'Not the highest card'],
                    ['pointCardNotAllowed', 'Not a point card'],
                  ]}
                  onChange={(v) => set('friendSelectionPolicy', v)}
                />
              </Row>
              <Row label="Landlord plays a called card">
                <Select
                  value={config.multipleJoinPolicy}
                  options={[
                    ['unrestricted', 'Uses up the call (team stays small)'],
                    ['noDoubleJoin', 'The next copy counts instead'],
                  ]}
                  onChange={(v) => set('multipleJoinPolicy', v)}
                />
              </Row>
              <Row label="Short-handed win bonus">
                <input type="checkbox" checked={config.bonusLevelForSmallerTeam} onChange={(e) => set('bonusLevelForSmallerTeam', e.target.checked)} />
              </Row>
            </>
          )}
          <Row label="Tractor needs" hint="Width × length of the shortest tractor.">
            <span className="opt-inline">
              <Select value={config.tractorMinCount} options={[[2, 'pairs'], [3, 'triples'], [4, 'quads']]} onChange={(v) => set('tractorMinCount', v)} />
              <span>×</span>
              <Select value={config.tractorMinLength} options={[[2, '2 long'], [3, '3 long'], [4, '4 long']]} onChange={(v) => set('tractorMinLength', v)} />
            </span>
          </Row>
          <Row label="Points per level" hint="Per deck; 20 is the classic 80-to-take-over with two decks.">
            <Select value={config.stepSizePerDeck} options={[[10, '10'], [15, '15'], [20, '20'], [25, '25'], [30, '30']]} onChange={(v) => set('stepSizePerDeck', v)} />
          </Row>
          <Row label="Show the kitty after a round">
            <input type="checkbox" checked={config.revealKittyAtEnd} onChange={(e) => set('revealKittyAtEnd', e.target.checked)} />
          </Row>
        </div>
      )}

      {errors.length > 0 ? (
        <div className="online-error">{errors.join(' ')}</div>
      ) : (
        <p className="lobby-hint opt-summary">{summary}</p>
      )}
    </div>
  );
}

function resolveKitty(config: TractorConfig, seats: readonly Seat[]): number {
  try {
    return resolveRules({ ...config, kittySize: null }, seats).kittySize;
  } catch {
    return 8;
  }
}
