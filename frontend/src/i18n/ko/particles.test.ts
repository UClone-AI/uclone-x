import { describe, it, expect } from 'vitest';
import { finalConsonant, koParticle, koWithParticle } from './particles';
import { fmt } from '../format';
import { ko } from './index';

describe('finalConsonant', () => {
  it('reads the 받침 of the last Hangul syllable, and nothing from anything else', () => {
    expect(finalConsonant('민지')).toBe(0);
    expect(finalConsonant('지훈')).toBe(4); // ㄴ
    expect(finalConsonant('서울')).toBe(8); // ㄹ
    expect(finalConsonant('가')).toBe(0);
    expect(finalConsonant('힣')).toBe(27);
    expect(finalConsonant('지훈  ')).toBe(4);
    expect(finalConsonant('Scout')).toBeNull();
    expect(finalConsonant('qwen3')).toBeNull();
    expect(finalConsonant('ㄱ')).toBeNull();
    expect(finalConsonant('')).toBeNull();
  });
});

describe('koParticle', () => {
  // Killed by: frontend/src/i18n/ko/particles.ts ::   return jong === 0 ? forms[1] : forms[0];
  // Becomes:   return jong !== 0 ? forms[1] : forms[0];
  it('picks the form after a 받침 and the form after none, for every pair the catalog writes', () => {
    expect([koParticle('지훈', '이(가)'), koParticle('민지', '이(가)')]).toEqual(['이', '가']);
    expect([koParticle('지훈', '을(를)'), koParticle('민지', '을(를)')]).toEqual(['을', '를']);
    expect([koParticle('지훈', '은(는)'), koParticle('민지', '은(는)')]).toEqual(['은', '는']);
    expect([koParticle('지훈', '와(과)'), koParticle('민지', '와(과)')]).toEqual(['과', '와']);
    expect([koParticle('지훈', '(으)로'), koParticle('민지', '(으)로')]).toEqual(['으로', '로']);
  });

  // Killed by: frontend/src/i18n/ko/particles.ts ::   if (pair === '(으)로' && jong === RIEUL) return forms[1];
  // Becomes:   if (pair === '(으)로' && jong === -1) return forms[1];
  it('takes 로, not 으로, after ㄹ', () => {
    expect(koParticle('서울', '(으)로')).toBe('로');
    expect(koParticle('서울', '이(가)')).toBe('이');
  });

  // Killed by: frontend/src/i18n/ko/particles.ts ::   if (!forms || jong === null) return pair;
  // Becomes:   if (!forms) return pair;
  it('keeps the paired form after a name that does not end in Hangul, by the author choice', () => {
    expect(koParticle('Scout', '이(가)')).toBe('이(가)');
    expect(koParticle('qwen3:8b', '을(를)')).toBe('을(를)');
    expect(koParticle('', '은(는)')).toBe('은(는)');
    expect(koParticle('지훈', '의')).toBe('의');
  });
});

describe('fmt with a Korean paired particle', () => {
  // Killed by: frontend/src/i18n/format.ts ::       return `${value}${quote}${particle ? koParticle(value, particle) : ''}`;
  // Becomes:       return `${value}${quote}${particle ?? ''}`;
  it('settles the particle from the value it follows, through a closing quote too', () => {
    expect(fmt(ko.conversation.membership.joined, { name: '민지' })).toBe('민지가 이 대화에 참여했습니다');
    expect(fmt(ko.conversation.membership.left, { name: '지훈' })).toBe('지훈이 이 대화에서 나갔습니다');
    expect(fmt(ko.settings.feedback.deleted, { model: '라마' })).toBe('모델 "라마"를 삭제했습니다.');
    expect(fmt(ko.images.now.drawing, { engine: '서울' })).toBe('지금은 서울로 그림을 그립니다.');
  });

  it('leaves the pair after a value that does not end in Hangul, and touches no English', () => {
    expect(fmt(ko.conversation.membership.joined, { name: 'Scout' })).toBe(
      'Scout이(가) 이 대화에 참여했습니다',
    );
    expect(fmt("{label}'s turn", { label: 'Scout' })).toBe("Scout's turn");
    expect(fmt('{a}{b}', { a: '지훈', b: '이(가)' })).toBe('지훈이(가)');
  });
});

describe('the pronouns before the subject particle', () => {
  // Killed by: frontend/src/i18n/ko/particles.ts ::   (pair === '이(가)' ? SUBJECT_PRONOUNS[word] : undefined) ?? `${word}${koParticle(word, pair)}`;
  // Becomes:   `${word}${koParticle(word, pair)}`;
  it('writes 내가, 제가 and 네가, never 나가, 저가 or 너가', () => {
    expect(koWithParticle('나', '이(가)')).toBe('내가');
    expect(koWithParticle('저', '이(가)')).toBe('제가');
    expect(koWithParticle('너', '이(가)')).toBe('네가');
    expect(fmt(ko.conversation.membership.joined, { name: '나' })).toBe('내가 이 대화에 참여했습니다');
  });

  // Killed by: frontend/src/i18n/ko/particles.ts ::   (pair === '이(가)' ? SUBJECT_PRONOUNS[word] : undefined) ?? `${word}${koParticle(word, pair)}`;
  // Becomes:   (SUBJECT_PRONOUNS[word] ?? `${word}${koParticle(word, pair)}`);
  it('changes only the subject particle, and only a value that is the pronoun itself', () => {
    expect(koWithParticle('나', '은(는)')).toBe('나는');
    expect(koWithParticle('나', '을(를)')).toBe('나를');
    expect(koWithParticle('저', '와(과)')).toBe('저와');
    expect(koWithParticle('하나', '이(가)')).toBe('하나가');
    expect(koWithParticle('지훈', '이(가)')).toBe('지훈이');
    expect(koWithParticle('Scout', '이(가)')).toBe('Scout이(가)');
  });

  // Killed by: frontend/src/i18n/format.ts ::       if (particle && !quote) return koWithParticle(value, particle);
  // Becomes:       if (particle) return koWithParticle(value, particle);
  it('leaves a quoted pronoun as the word it quotes', () => {
    expect(fmt('"{name}"이(가) 없습니다', { name: '나' })).toBe('"나"가 없습니다');
  });
});
