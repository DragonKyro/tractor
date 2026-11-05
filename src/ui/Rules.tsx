// The rules, shown from the lobby and in game. The scoring table is
// computed from the same code the game scores with.

import { explainScoring, materialize, DEFAULT_SCORING } from '@/game/scoring';

function ScoringTable({ decks }: { decks: number }) {
  const m = materialize(DEFAULT_SCORING, decks, decks * 100);
  const rows = explainScoring(m);
  return (
    <table className="rules-table">
      <thead>
        <tr>
          <th>Attackers' points</th>
          <th>Result</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => {
          const to = rows[i + 1] ? `–${rows[i + 1]!.from - 5}` : '+';
          const s = r.score;
          return (
            <tr key={r.from}>
              <td>{r.from}{to}</td>
              <td>
                {s.landlordWon
                  ? `Landlord's team up ${s.landlordDelta}`
                  : s.attackerDelta === 0
                    ? 'Attackers take over (no level)'
                    : `Attackers take over, up ${s.attackerDelta}`}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function RulesText() {
  return (
    <div className="rules-text">
      <h4>The idea</h4>
      <p>
        Tractor (升级, also <em>shengji</em>, “80 points”) is a trick-taking game for teams. Each player has a <strong>level</strong>{' '}
        (a rank, starting at 2). Every round one side <strong>defends</strong> — the landlord and their team — and the other side{' '}
        <strong>attacks</strong>, trying to capture points: every 5 is worth 5, every 10 and K is worth 10 (100 per deck). Win as
        defenders to climb levels; collect enough as attackers to take over. The first to play past the final rank (A by default)
        wins.
      </p>

      <h4>Teams</h4>
      <p>
        <strong>Tractor</strong>: fixed partners sitting opposite each other (even tables). <strong>Finding Friends</strong>: the
        landlord secretly calls partners by naming a card (“the second A♠”); whoever plays that copy joins the landlord's team —
        nobody knows for sure until then. Odd-sized tables play Finding Friends.
      </p>

      <h4>Trump</h4>
      <p>
        The round's <strong>trump number</strong> is the landlord's level. Every card of that number, both jokers, and every card of
        the <strong>trump suit</strong> are trump. Trump beats everything else. Within trump, from low to high: the trump suit 2…A
        (skipping the number), the number in the other suits, the number in the trump suit, the small joker, the big joker.
      </p>

      <h4>Dealing and declaring</h4>
      <p>
        Cards are dealt out; a few stay face down as the <strong>kitty</strong>. While the deal goes on, show a card of the trump
        number to <strong>declare</strong> its suit as trump. Someone else can overcall with more copies (a pair beats a single), and
        a pair of jokers declares no trump suit (only numbers and jokers are trump). You may add copies to your own bid to defend it.
        In the first round, whoever's declaration stands becomes the landlord. If nobody declares, the deal is redone (or, once
        there's a landlord, kitty cards are turned over to set trump). On a shared screen, bidding instead goes round the table
        after the deal.
      </p>

      <h4>The kitty</h4>
      <p>
        The landlord picks up the kitty and buries the same number of cards. Points buried there go to the attackers if they win the
        last trick — multiplied by 2 × the size of the largest unit in that trick (a pair doubles it again).
      </p>

      <h4>Leading</h4>
      <p>A lead is all one suit (or all trump) and can be:</p>
      <ul>
        <li><strong>a single</strong>, <strong>a pair</strong> (two identical cards), a triple…</li>
        <li>
          <strong>a tractor</strong>: consecutive pairs in one suit, like 7♥7♥8♥8♥. The trump number is skipped (with 9 trump,
          8♥8♥10♥10♥ is a tractor), and in trump the ladder runs on through the numbers and jokers.
        </li>
        <li>
          <strong>a throw</strong>: several of the above at once — allowed only if nobody can beat any part of it. If someone can, only
          the beatable part is played and the rest goes back to your hand.
        </li>
      </ul>

      <h4>Following</h4>
      <p>
        Play as many cards as were led. Follow the led suit if you can; if you have enough of it you must also match its{' '}
        <em>shape</em> as far as your hand allows — pairs on a pair, a tractor on a tractor, falling back to pairs, then singles. Out of
        the suit, play anything — trump that matches the lead's shape can win it.
      </p>

      <h4>Winning a trick</h4>
      <p>
        The highest card (or unit) of the led suit wins, unless the trick was trumped — then the highest matching trump play wins. A
        play only counts if it matches the lead's shape. The winner takes the trick's points and leads next.
      </p>

      <h4>Scoring (two decks)</h4>
      <ScoringTable decks={2} />
      <p className="lobby-hint">
        With other deck counts the steps scale (20 points per deck). Levels stop at the final rank: to pass it you must win it as
        defenders. A landlord's team that wins short-handed in Finding Friends gets a bonus level.
      </p>

      <h4>Next round</h4>
      <p>
        If the defenders hold, the next defender after the landlord (their partner in Tractor) becomes landlord; if the attackers take
        over, the next attacker after the landlord does.
      </p>

      <p className="lobby-hint">
        Rules follow the open-source <em>shengji</em> implementation (robertying.com/shengji); its house-rule options are under “More
        options” in the lobby.
      </p>
    </div>
  );
}
