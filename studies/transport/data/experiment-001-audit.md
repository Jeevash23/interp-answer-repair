# Experiment 001 structured prompt audit

First 100 manifest rows, generated before model outcome evaluation.

## 001 — exploratory-000-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Peter`, subject=`Anna`, corruption=`John`
- canonical clean: `When Peter and Anna went to the store, Anna gave a drink to`
- canonical corrupt: `When Peter and Anna went to the store, John gave a drink to`
- controlled clean: `Thus Peter beside Anna painted every silent garden; Anna carried some token toward`
- controlled corrupt: `Thus Peter beside Anna painted every silent garden; John carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 002 — exploratory-000-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Peter`, subject=`Anna`, corruption=`John`
- canonical clean: `When Anna and Peter went to the store, Anna gave a drink to`
- canonical corrupt: `When Anna and Peter went to the store, John gave a drink to`
- controlled clean: `Thus Anna beside Peter painted every silent garden; Anna carried some token toward`
- controlled corrupt: `Thus Anna beside Peter painted every silent garden; John carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 003 — exploratory-001-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Margaret`, subject=`Laura`, corruption=`Charles`
- canonical clean: `Then Margaret and Laura went to the park, Laura handed a book to`
- canonical corrupt: `Then Margaret and Laura went to the park, Charles handed a book to`
- controlled clean: `Today Margaret near Laura crossed each quiet field; Laura moved one object toward`
- controlled corrupt: `Today Margaret near Laura crossed each quiet field; Charles moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 004 — exploratory-001-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Margaret`, subject=`Laura`, corruption=`Charles`
- canonical clean: `Then Laura and Margaret went to the park, Laura handed a book to`
- canonical corrupt: `Then Laura and Margaret went to the park, Charles handed a book to`
- controlled clean: `Today Laura near Margaret crossed each quiet field; Laura moved one object toward`
- controlled corrupt: `Today Laura near Margaret crossed each quiet field; Charles moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 005 — exploratory-002-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Richard`, subject=`Anne`, corruption=`David`
- canonical clean: `When Richard and Anne went to the store, Anne gave a drink to`
- canonical corrupt: `When Richard and Anne went to the store, David gave a drink to`
- controlled clean: `Thus Richard beside Anne painted every silent garden; Anne carried some token toward`
- controlled corrupt: `Thus Richard beside Anne painted every silent garden; David carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 006 — exploratory-002-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Richard`, subject=`Anne`, corruption=`David`
- canonical clean: `When Anne and Richard went to the store, Anne gave a drink to`
- canonical corrupt: `When Anne and Richard went to the store, David gave a drink to`
- controlled clean: `Thus Anne beside Richard painted every silent garden; Anne carried some token toward`
- controlled corrupt: `Thus Anne beside Richard painted every silent garden; David carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 007 — exploratory-003-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`George`, subject=`Mark`, corruption=`Jason`
- canonical clean: `Then George and Mark went to the park, Mark handed a book to`
- canonical corrupt: `Then George and Mark went to the park, Jason handed a book to`
- controlled clean: `Today George near Mark crossed each quiet field; Mark moved one object toward`
- controlled corrupt: `Today George near Mark crossed each quiet field; Jason moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 008 — exploratory-003-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`George`, subject=`Mark`, corruption=`Jason`
- canonical clean: `Then Mark and George went to the park, Mark handed a book to`
- canonical corrupt: `Then Mark and George went to the park, Jason handed a book to`
- controlled clean: `Today Mark near George crossed each quiet field; Mark moved one object toward`
- controlled corrupt: `Today Mark near George crossed each quiet field; Jason moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 009 — exploratory-004-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`George`, subject=`Edward`, corruption=`Carol`
- canonical clean: `When George and Edward went to the store, Edward gave a drink to`
- canonical corrupt: `When George and Edward went to the store, Carol gave a drink to`
- controlled clean: `Thus George beside Edward painted every silent garden; Edward carried some token toward`
- controlled corrupt: `Thus George beside Edward painted every silent garden; Carol carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 010 — exploratory-004-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`George`, subject=`Edward`, corruption=`Carol`
- canonical clean: `When Edward and George went to the store, Edward gave a drink to`
- canonical corrupt: `When Edward and George went to the store, Carol gave a drink to`
- controlled clean: `Thus Edward beside George painted every silent garden; Edward carried some token toward`
- controlled corrupt: `Thus Edward beside George painted every silent garden; Carol carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 011 — exploratory-005-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Sarah`, subject=`Mary`, corruption=`Lisa`
- canonical clean: `Then Sarah and Mary went to the park, Mary handed a book to`
- canonical corrupt: `Then Sarah and Mary went to the park, Lisa handed a book to`
- controlled clean: `Today Sarah near Mary crossed each quiet field; Mary moved one object toward`
- controlled corrupt: `Today Sarah near Mary crossed each quiet field; Lisa moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 012 — exploratory-005-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Sarah`, subject=`Mary`, corruption=`Lisa`
- canonical clean: `Then Mary and Sarah went to the park, Mary handed a book to`
- canonical corrupt: `Then Mary and Sarah went to the park, Lisa handed a book to`
- controlled clean: `Today Mary near Sarah crossed each quiet field; Mary moved one object toward`
- controlled corrupt: `Today Mary near Sarah crossed each quiet field; Lisa moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 013 — exploratory-006-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Alice`, subject=`Patricia`, corruption=`Laura`
- canonical clean: `When Alice and Patricia went to the store, Patricia gave a drink to`
- canonical corrupt: `When Alice and Patricia went to the store, Laura gave a drink to`
- controlled clean: `Thus Alice beside Patricia painted every silent garden; Patricia carried some token toward`
- controlled corrupt: `Thus Alice beside Patricia painted every silent garden; Laura carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 014 — exploratory-006-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Alice`, subject=`Patricia`, corruption=`Laura`
- canonical clean: `When Patricia and Alice went to the store, Patricia gave a drink to`
- canonical corrupt: `When Patricia and Alice went to the store, Laura gave a drink to`
- controlled clean: `Thus Patricia beside Alice painted every silent garden; Patricia carried some token toward`
- controlled corrupt: `Thus Patricia beside Alice painted every silent garden; Laura carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 015 — exploratory-007-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Jason`, subject=`Jessica`, corruption=`James`
- canonical clean: `Then Jason and Jessica went to the park, Jessica handed a book to`
- canonical corrupt: `Then Jason and Jessica went to the park, James handed a book to`
- controlled clean: `Today Jason near Jessica crossed each quiet field; Jessica moved one object toward`
- controlled corrupt: `Today Jason near Jessica crossed each quiet field; James moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 016 — exploratory-007-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Jason`, subject=`Jessica`, corruption=`James`
- canonical clean: `Then Jessica and Jason went to the park, Jessica handed a book to`
- canonical corrupt: `Then Jessica and Jason went to the park, James handed a book to`
- controlled clean: `Today Jessica near Jason crossed each quiet field; Jessica moved one object toward`
- controlled corrupt: `Today Jessica near Jason crossed each quiet field; James moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 017 — exploratory-008-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Sharon`, subject=`Robert`, corruption=`Sandra`
- canonical clean: `When Sharon and Robert went to the store, Robert gave a drink to`
- canonical corrupt: `When Sharon and Robert went to the store, Sandra gave a drink to`
- controlled clean: `Thus Sharon beside Robert painted every silent garden; Robert carried some token toward`
- controlled corrupt: `Thus Sharon beside Robert painted every silent garden; Sandra carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 018 — exploratory-008-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Sharon`, subject=`Robert`, corruption=`Sandra`
- canonical clean: `When Robert and Sharon went to the store, Robert gave a drink to`
- canonical corrupt: `When Robert and Sharon went to the store, Sandra gave a drink to`
- controlled clean: `Thus Robert beside Sharon painted every silent garden; Robert carried some token toward`
- controlled corrupt: `Thus Robert beside Sharon painted every silent garden; Sandra carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 019 — exploratory-009-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Susan`, subject=`John`, corruption=`Patricia`
- canonical clean: `Then Susan and John went to the park, John handed a book to`
- canonical corrupt: `Then Susan and John went to the park, Patricia handed a book to`
- controlled clean: `Today Susan near John crossed each quiet field; John moved one object toward`
- controlled corrupt: `Today Susan near John crossed each quiet field; Patricia moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 020 — exploratory-009-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Susan`, subject=`John`, corruption=`Patricia`
- canonical clean: `Then John and Susan went to the park, John handed a book to`
- canonical corrupt: `Then John and Susan went to the park, Patricia handed a book to`
- controlled clean: `Today John near Susan crossed each quiet field; John moved one object toward`
- controlled corrupt: `Today John near Susan crossed each quiet field; Patricia moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 021 — exploratory-010-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`James`, subject=`Mary`, corruption=`Elizabeth`
- canonical clean: `When James and Mary went to the store, Mary gave a drink to`
- canonical corrupt: `When James and Mary went to the store, Elizabeth gave a drink to`
- controlled clean: `Thus James beside Mary painted every silent garden; Mary carried some token toward`
- controlled corrupt: `Thus James beside Mary painted every silent garden; Elizabeth carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 022 — exploratory-010-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`James`, subject=`Mary`, corruption=`Elizabeth`
- canonical clean: `When Mary and James went to the store, Mary gave a drink to`
- canonical corrupt: `When Mary and James went to the store, Elizabeth gave a drink to`
- controlled clean: `Thus Mary beside James painted every silent garden; Mary carried some token toward`
- controlled corrupt: `Thus Mary beside James painted every silent garden; Elizabeth carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 023 — exploratory-011-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Jessica`, subject=`Anne`, corruption=`Michael`
- canonical clean: `Then Jessica and Anne went to the park, Anne handed a book to`
- canonical corrupt: `Then Jessica and Anne went to the park, Michael handed a book to`
- controlled clean: `Today Jessica near Anne crossed each quiet field; Anne moved one object toward`
- controlled corrupt: `Today Jessica near Anne crossed each quiet field; Michael moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 024 — exploratory-011-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Jessica`, subject=`Anne`, corruption=`Michael`
- canonical clean: `Then Anne and Jessica went to the park, Anne handed a book to`
- canonical corrupt: `Then Anne and Jessica went to the park, Michael handed a book to`
- controlled clean: `Today Anne near Jessica crossed each quiet field; Anne moved one object toward`
- controlled corrupt: `Today Anne near Jessica crossed each quiet field; Michael moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 025 — exploratory-012-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Anne`, subject=`Sandra`, corruption=`Nancy`
- canonical clean: `When Anne and Sandra went to the store, Sandra gave a drink to`
- canonical corrupt: `When Anne and Sandra went to the store, Nancy gave a drink to`
- controlled clean: `Thus Anne beside Sandra painted every silent garden; Sandra carried some token toward`
- controlled corrupt: `Thus Anne beside Sandra painted every silent garden; Nancy carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 026 — exploratory-012-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Anne`, subject=`Sandra`, corruption=`Nancy`
- canonical clean: `When Sandra and Anne went to the store, Sandra gave a drink to`
- canonical corrupt: `When Sandra and Anne went to the store, Nancy gave a drink to`
- controlled clean: `Thus Sandra beside Anne painted every silent garden; Sandra carried some token toward`
- controlled corrupt: `Thus Sandra beside Anne painted every silent garden; Nancy carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 027 — exploratory-013-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Betty`, subject=`Martin`, corruption=`Jennifer`
- canonical clean: `Then Betty and Martin went to the park, Martin handed a book to`
- canonical corrupt: `Then Betty and Martin went to the park, Jennifer handed a book to`
- controlled clean: `Today Betty near Martin crossed each quiet field; Martin moved one object toward`
- controlled corrupt: `Today Betty near Martin crossed each quiet field; Jennifer moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 028 — exploratory-013-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Betty`, subject=`Martin`, corruption=`Jennifer`
- canonical clean: `Then Martin and Betty went to the park, Martin handed a book to`
- canonical corrupt: `Then Martin and Betty went to the park, Jennifer handed a book to`
- controlled clean: `Today Martin near Betty crossed each quiet field; Martin moved one object toward`
- controlled corrupt: `Today Martin near Betty crossed each quiet field; Jennifer moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 029 — exploratory-014-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Sarah`, subject=`Nancy`, corruption=`Michael`
- canonical clean: `When Sarah and Nancy went to the store, Nancy gave a drink to`
- canonical corrupt: `When Sarah and Nancy went to the store, Michael gave a drink to`
- controlled clean: `Thus Sarah beside Nancy painted every silent garden; Nancy carried some token toward`
- controlled corrupt: `Thus Sarah beside Nancy painted every silent garden; Michael carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 030 — exploratory-014-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Sarah`, subject=`Nancy`, corruption=`Michael`
- canonical clean: `When Nancy and Sarah went to the store, Nancy gave a drink to`
- canonical corrupt: `When Nancy and Sarah went to the store, Michael gave a drink to`
- controlled clean: `Thus Nancy beside Sarah painted every silent garden; Nancy carried some token toward`
- controlled corrupt: `Thus Nancy beside Sarah painted every silent garden; Michael carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 031 — exploratory-015-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Walter`, subject=`Helen`, corruption=`James`
- canonical clean: `Then Walter and Helen went to the park, Helen handed a book to`
- canonical corrupt: `Then Walter and Helen went to the park, James handed a book to`
- controlled clean: `Today Walter near Helen crossed each quiet field; Helen moved one object toward`
- controlled corrupt: `Today Walter near Helen crossed each quiet field; James moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 032 — exploratory-015-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Walter`, subject=`Helen`, corruption=`James`
- canonical clean: `Then Helen and Walter went to the park, Helen handed a book to`
- canonical corrupt: `Then Helen and Walter went to the park, James handed a book to`
- controlled clean: `Today Helen near Walter crossed each quiet field; Helen moved one object toward`
- controlled corrupt: `Today Helen near Walter crossed each quiet field; James moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 033 — exploratory-016-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`William`, subject=`Alice`, corruption=`Michael`
- canonical clean: `When William and Alice went to the store, Alice gave a drink to`
- canonical corrupt: `When William and Alice went to the store, Michael gave a drink to`
- controlled clean: `Thus William beside Alice painted every silent garden; Alice carried some token toward`
- controlled corrupt: `Thus William beside Alice painted every silent garden; Michael carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 034 — exploratory-016-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`William`, subject=`Alice`, corruption=`Michael`
- canonical clean: `When Alice and William went to the store, Alice gave a drink to`
- canonical corrupt: `When Alice and William went to the store, Michael gave a drink to`
- controlled clean: `Thus Alice beside William painted every silent garden; Alice carried some token toward`
- controlled corrupt: `Thus Alice beside William painted every silent garden; Michael carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 035 — exploratory-017-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`George`, subject=`Michael`, corruption=`Laura`
- canonical clean: `Then George and Michael went to the park, Michael handed a book to`
- canonical corrupt: `Then George and Michael went to the park, Laura handed a book to`
- controlled clean: `Today George near Michael crossed each quiet field; Michael moved one object toward`
- controlled corrupt: `Today George near Michael crossed each quiet field; Laura moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 036 — exploratory-017-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`George`, subject=`Michael`, corruption=`Laura`
- canonical clean: `Then Michael and George went to the park, Michael handed a book to`
- canonical corrupt: `Then Michael and George went to the park, Laura handed a book to`
- controlled clean: `Today Michael near George crossed each quiet field; Michael moved one object toward`
- controlled corrupt: `Today Michael near George crossed each quiet field; Laura moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 037 — exploratory-018-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Michelle`, subject=`Linda`, corruption=`John`
- canonical clean: `When Michelle and Linda went to the store, Linda gave a drink to`
- canonical corrupt: `When Michelle and Linda went to the store, John gave a drink to`
- controlled clean: `Thus Michelle beside Linda painted every silent garden; Linda carried some token toward`
- controlled corrupt: `Thus Michelle beside Linda painted every silent garden; John carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 038 — exploratory-018-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Michelle`, subject=`Linda`, corruption=`John`
- canonical clean: `When Linda and Michelle went to the store, Linda gave a drink to`
- canonical corrupt: `When Linda and Michelle went to the store, John gave a drink to`
- controlled clean: `Thus Linda beside Michelle painted every silent garden; Linda carried some token toward`
- controlled corrupt: `Thus Linda beside Michelle painted every silent garden; John carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 039 — exploratory-019-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Kevin`, subject=`Karen`, corruption=`Elizabeth`
- canonical clean: `Then Kevin and Karen went to the park, Karen handed a book to`
- canonical corrupt: `Then Kevin and Karen went to the park, Elizabeth handed a book to`
- controlled clean: `Today Kevin near Karen crossed each quiet field; Karen moved one object toward`
- controlled corrupt: `Today Kevin near Karen crossed each quiet field; Elizabeth moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 040 — exploratory-019-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Kevin`, subject=`Karen`, corruption=`Elizabeth`
- canonical clean: `Then Karen and Kevin went to the park, Karen handed a book to`
- canonical corrupt: `Then Karen and Kevin went to the park, Elizabeth handed a book to`
- controlled clean: `Today Karen near Kevin crossed each quiet field; Karen moved one object toward`
- controlled corrupt: `Today Karen near Kevin crossed each quiet field; Elizabeth moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 041 — exploratory-020-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Alice`, subject=`Gary`, corruption=`Jessica`
- canonical clean: `When Alice and Gary went to the store, Gary gave a drink to`
- canonical corrupt: `When Alice and Gary went to the store, Jessica gave a drink to`
- controlled clean: `Thus Alice beside Gary painted every silent garden; Gary carried some token toward`
- controlled corrupt: `Thus Alice beside Gary painted every silent garden; Jessica carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 042 — exploratory-020-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Alice`, subject=`Gary`, corruption=`Jessica`
- canonical clean: `When Gary and Alice went to the store, Gary gave a drink to`
- canonical corrupt: `When Gary and Alice went to the store, Jessica gave a drink to`
- controlled clean: `Thus Gary beside Alice painted every silent garden; Gary carried some token toward`
- controlled corrupt: `Thus Gary beside Alice painted every silent garden; Jessica carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 043 — exploratory-021-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Bob`, subject=`Edward`, corruption=`David`
- canonical clean: `Then Bob and Edward went to the park, Edward handed a book to`
- canonical corrupt: `Then Bob and Edward went to the park, David handed a book to`
- controlled clean: `Today Bob near Edward crossed each quiet field; Edward moved one object toward`
- controlled corrupt: `Today Bob near Edward crossed each quiet field; David moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 044 — exploratory-021-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Bob`, subject=`Edward`, corruption=`David`
- canonical clean: `Then Edward and Bob went to the park, Edward handed a book to`
- canonical corrupt: `Then Edward and Bob went to the park, David handed a book to`
- controlled clean: `Today Edward near Bob crossed each quiet field; Edward moved one object toward`
- controlled corrupt: `Today Edward near Bob crossed each quiet field; David moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 045 — exploratory-022-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Laura`, subject=`George`, corruption=`Mary`
- canonical clean: `When Laura and George went to the store, George gave a drink to`
- canonical corrupt: `When Laura and George went to the store, Mary gave a drink to`
- controlled clean: `Thus Laura beside George painted every silent garden; George carried some token toward`
- controlled corrupt: `Thus Laura beside George painted every silent garden; Mary carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 046 — exploratory-022-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Laura`, subject=`George`, corruption=`Mary`
- canonical clean: `When George and Laura went to the store, George gave a drink to`
- canonical corrupt: `When George and Laura went to the store, Mary gave a drink to`
- controlled clean: `Thus George beside Laura painted every silent garden; George carried some token toward`
- controlled corrupt: `Thus George beside Laura painted every silent garden; Mary carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 047 — exploratory-023-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Laura`, subject=`Betty`, corruption=`Ruth`
- canonical clean: `Then Laura and Betty went to the park, Betty handed a book to`
- canonical corrupt: `Then Laura and Betty went to the park, Ruth handed a book to`
- controlled clean: `Today Laura near Betty crossed each quiet field; Betty moved one object toward`
- controlled corrupt: `Today Laura near Betty crossed each quiet field; Ruth moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 048 — exploratory-023-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Laura`, subject=`Betty`, corruption=`Ruth`
- canonical clean: `Then Betty and Laura went to the park, Betty handed a book to`
- canonical corrupt: `Then Betty and Laura went to the park, Ruth handed a book to`
- controlled clean: `Today Betty near Laura crossed each quiet field; Betty moved one object toward`
- controlled corrupt: `Today Betty near Laura crossed each quiet field; Ruth moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 049 — exploratory-024-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Kevin`, subject=`Carol`, corruption=`Peter`
- canonical clean: `When Kevin and Carol went to the store, Carol gave a drink to`
- canonical corrupt: `When Kevin and Carol went to the store, Peter gave a drink to`
- controlled clean: `Thus Kevin beside Carol painted every silent garden; Carol carried some token toward`
- controlled corrupt: `Thus Kevin beside Carol painted every silent garden; Peter carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 050 — exploratory-024-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Kevin`, subject=`Carol`, corruption=`Peter`
- canonical clean: `When Carol and Kevin went to the store, Carol gave a drink to`
- canonical corrupt: `When Carol and Kevin went to the store, Peter gave a drink to`
- controlled clean: `Thus Carol beside Kevin painted every silent garden; Carol carried some token toward`
- controlled corrupt: `Thus Carol beside Kevin painted every silent garden; Peter carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 051 — exploratory-025-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Jason`, subject=`Gary`, corruption=`Mark`
- canonical clean: `Then Jason and Gary went to the park, Gary handed a book to`
- canonical corrupt: `Then Jason and Gary went to the park, Mark handed a book to`
- controlled clean: `Today Jason near Gary crossed each quiet field; Gary moved one object toward`
- controlled corrupt: `Today Jason near Gary crossed each quiet field; Mark moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 052 — exploratory-025-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Jason`, subject=`Gary`, corruption=`Mark`
- canonical clean: `Then Gary and Jason went to the park, Gary handed a book to`
- canonical corrupt: `Then Gary and Jason went to the park, Mark handed a book to`
- controlled clean: `Today Gary near Jason crossed each quiet field; Gary moved one object toward`
- controlled corrupt: `Today Gary near Jason crossed each quiet field; Mark moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 053 — exploratory-026-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Kevin`, subject=`Edward`, corruption=`Henry`
- canonical clean: `When Kevin and Edward went to the store, Edward gave a drink to`
- canonical corrupt: `When Kevin and Edward went to the store, Henry gave a drink to`
- controlled clean: `Thus Kevin beside Edward painted every silent garden; Edward carried some token toward`
- controlled corrupt: `Thus Kevin beside Edward painted every silent garden; Henry carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 054 — exploratory-026-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Kevin`, subject=`Edward`, corruption=`Henry`
- canonical clean: `When Edward and Kevin went to the store, Edward gave a drink to`
- canonical corrupt: `When Edward and Kevin went to the store, Henry gave a drink to`
- controlled clean: `Thus Edward beside Kevin painted every silent garden; Edward carried some token toward`
- controlled corrupt: `Thus Edward beside Kevin painted every silent garden; Henry carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 055 — exploratory-027-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Bob`, subject=`Anne`, corruption=`Sarah`
- canonical clean: `Then Bob and Anne went to the park, Anne handed a book to`
- canonical corrupt: `Then Bob and Anne went to the park, Sarah handed a book to`
- controlled clean: `Today Bob near Anne crossed each quiet field; Anne moved one object toward`
- controlled corrupt: `Today Bob near Anne crossed each quiet field; Sarah moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 056 — exploratory-027-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Bob`, subject=`Anne`, corruption=`Sarah`
- canonical clean: `Then Anne and Bob went to the park, Anne handed a book to`
- canonical corrupt: `Then Anne and Bob went to the park, Sarah handed a book to`
- controlled clean: `Today Anne near Bob crossed each quiet field; Anne moved one object toward`
- controlled corrupt: `Today Anne near Bob crossed each quiet field; Sarah moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 057 — exploratory-028-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Mark`, subject=`Paul`, corruption=`Lisa`
- canonical clean: `When Mark and Paul went to the store, Paul gave a drink to`
- canonical corrupt: `When Mark and Paul went to the store, Lisa gave a drink to`
- controlled clean: `Thus Mark beside Paul painted every silent garden; Paul carried some token toward`
- controlled corrupt: `Thus Mark beside Paul painted every silent garden; Lisa carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 058 — exploratory-028-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Mark`, subject=`Paul`, corruption=`Lisa`
- canonical clean: `When Paul and Mark went to the store, Paul gave a drink to`
- canonical corrupt: `When Paul and Mark went to the store, Lisa gave a drink to`
- controlled clean: `Thus Paul beside Mark painted every silent garden; Paul carried some token toward`
- controlled corrupt: `Thus Paul beside Mark painted every silent garden; Lisa carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 059 — exploratory-029-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Anne`, subject=`George`, corruption=`Jason`
- canonical clean: `Then Anne and George went to the park, George handed a book to`
- canonical corrupt: `Then Anne and George went to the park, Jason handed a book to`
- controlled clean: `Today Anne near George crossed each quiet field; George moved one object toward`
- controlled corrupt: `Today Anne near George crossed each quiet field; Jason moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 060 — exploratory-029-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Anne`, subject=`George`, corruption=`Jason`
- canonical clean: `Then George and Anne went to the park, George handed a book to`
- canonical corrupt: `Then George and Anne went to the park, Jason handed a book to`
- controlled clean: `Today George near Anne crossed each quiet field; George moved one object toward`
- controlled corrupt: `Today George near Anne crossed each quiet field; Jason moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 061 — exploratory-030-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`John`, subject=`David`, corruption=`Michelle`
- canonical clean: `When John and David went to the store, David gave a drink to`
- canonical corrupt: `When John and David went to the store, Michelle gave a drink to`
- controlled clean: `Thus John beside David painted every silent garden; David carried some token toward`
- controlled corrupt: `Thus John beside David painted every silent garden; Michelle carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 062 — exploratory-030-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`John`, subject=`David`, corruption=`Michelle`
- canonical clean: `When David and John went to the store, David gave a drink to`
- canonical corrupt: `When David and John went to the store, Michelle gave a drink to`
- controlled clean: `Thus David beside John painted every silent garden; David carried some token toward`
- controlled corrupt: `Thus David beside John painted every silent garden; Michelle carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 063 — exploratory-031-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Charles`, subject=`Sandra`, corruption=`Mary`
- canonical clean: `Then Charles and Sandra went to the park, Sandra handed a book to`
- canonical corrupt: `Then Charles and Sandra went to the park, Mary handed a book to`
- controlled clean: `Today Charles near Sandra crossed each quiet field; Sandra moved one object toward`
- controlled corrupt: `Today Charles near Sandra crossed each quiet field; Mary moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 064 — exploratory-031-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Charles`, subject=`Sandra`, corruption=`Mary`
- canonical clean: `Then Sandra and Charles went to the park, Sandra handed a book to`
- canonical corrupt: `Then Sandra and Charles went to the park, Mary handed a book to`
- controlled clean: `Today Sandra near Charles crossed each quiet field; Sandra moved one object toward`
- controlled corrupt: `Today Sandra near Charles crossed each quiet field; Mary moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 065 — exploratory-032-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Laura`, subject=`Kevin`, corruption=`David`
- canonical clean: `When Laura and Kevin went to the store, Kevin gave a drink to`
- canonical corrupt: `When Laura and Kevin went to the store, David gave a drink to`
- controlled clean: `Thus Laura beside Kevin painted every silent garden; Kevin carried some token toward`
- controlled corrupt: `Thus Laura beside Kevin painted every silent garden; David carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 066 — exploratory-032-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Laura`, subject=`Kevin`, corruption=`David`
- canonical clean: `When Kevin and Laura went to the store, Kevin gave a drink to`
- canonical corrupt: `When Kevin and Laura went to the store, David gave a drink to`
- controlled clean: `Thus Kevin beside Laura painted every silent garden; Kevin carried some token toward`
- controlled corrupt: `Thus Kevin beside Laura painted every silent garden; David carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 067 — exploratory-033-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Kevin`, subject=`Sarah`, corruption=`Mary`
- canonical clean: `Then Kevin and Sarah went to the park, Sarah handed a book to`
- canonical corrupt: `Then Kevin and Sarah went to the park, Mary handed a book to`
- controlled clean: `Today Kevin near Sarah crossed each quiet field; Sarah moved one object toward`
- controlled corrupt: `Today Kevin near Sarah crossed each quiet field; Mary moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 068 — exploratory-033-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Kevin`, subject=`Sarah`, corruption=`Mary`
- canonical clean: `Then Sarah and Kevin went to the park, Sarah handed a book to`
- canonical corrupt: `Then Sarah and Kevin went to the park, Mary handed a book to`
- controlled clean: `Today Sarah near Kevin crossed each quiet field; Sarah moved one object toward`
- controlled corrupt: `Today Sarah near Kevin crossed each quiet field; Mary moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 069 — exploratory-034-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Frank`, subject=`Margaret`, corruption=`Laura`
- canonical clean: `When Frank and Margaret went to the store, Margaret gave a drink to`
- canonical corrupt: `When Frank and Margaret went to the store, Laura gave a drink to`
- controlled clean: `Thus Frank beside Margaret painted every silent garden; Margaret carried some token toward`
- controlled corrupt: `Thus Frank beside Margaret painted every silent garden; Laura carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 070 — exploratory-034-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Frank`, subject=`Margaret`, corruption=`Laura`
- canonical clean: `When Margaret and Frank went to the store, Margaret gave a drink to`
- canonical corrupt: `When Margaret and Frank went to the store, Laura gave a drink to`
- controlled clean: `Thus Margaret beside Frank painted every silent garden; Margaret carried some token toward`
- controlled corrupt: `Thus Margaret beside Frank painted every silent garden; Laura carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 071 — exploratory-035-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Margaret`, subject=`Sarah`, corruption=`Martin`
- canonical clean: `Then Margaret and Sarah went to the park, Sarah handed a book to`
- canonical corrupt: `Then Margaret and Sarah went to the park, Martin handed a book to`
- controlled clean: `Today Margaret near Sarah crossed each quiet field; Sarah moved one object toward`
- controlled corrupt: `Today Margaret near Sarah crossed each quiet field; Martin moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 072 — exploratory-035-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Margaret`, subject=`Sarah`, corruption=`Martin`
- canonical clean: `Then Sarah and Margaret went to the park, Sarah handed a book to`
- canonical corrupt: `Then Sarah and Margaret went to the park, Martin handed a book to`
- controlled clean: `Today Sarah near Margaret crossed each quiet field; Sarah moved one object toward`
- controlled corrupt: `Today Sarah near Margaret crossed each quiet field; Martin moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 073 — exploratory-036-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Nancy`, subject=`Ruth`, corruption=`Richard`
- canonical clean: `When Nancy and Ruth went to the store, Ruth gave a drink to`
- canonical corrupt: `When Nancy and Ruth went to the store, Richard gave a drink to`
- controlled clean: `Thus Nancy beside Ruth painted every silent garden; Ruth carried some token toward`
- controlled corrupt: `Thus Nancy beside Ruth painted every silent garden; Richard carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 074 — exploratory-036-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Nancy`, subject=`Ruth`, corruption=`Richard`
- canonical clean: `When Ruth and Nancy went to the store, Ruth gave a drink to`
- canonical corrupt: `When Ruth and Nancy went to the store, Richard gave a drink to`
- controlled clean: `Thus Ruth beside Nancy painted every silent garden; Ruth carried some token toward`
- controlled corrupt: `Thus Ruth beside Nancy painted every silent garden; Richard carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 075 — exploratory-037-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Bob`, subject=`Frank`, corruption=`Karen`
- canonical clean: `Then Bob and Frank went to the park, Frank handed a book to`
- canonical corrupt: `Then Bob and Frank went to the park, Karen handed a book to`
- controlled clean: `Today Bob near Frank crossed each quiet field; Frank moved one object toward`
- controlled corrupt: `Today Bob near Frank crossed each quiet field; Karen moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 076 — exploratory-037-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Bob`, subject=`Frank`, corruption=`Karen`
- canonical clean: `Then Frank and Bob went to the park, Frank handed a book to`
- canonical corrupt: `Then Frank and Bob went to the park, Karen handed a book to`
- controlled clean: `Today Frank near Bob crossed each quiet field; Frank moved one object toward`
- controlled corrupt: `Today Frank near Bob crossed each quiet field; Karen moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 077 — exploratory-038-abba

- split/template/order: `exploratory` / `explore-store` / `ABBA`
- roles: IO=`Sandra`, subject=`Betty`, corruption=`Arthur`
- canonical clean: `When Sandra and Betty went to the store, Betty gave a drink to`
- canonical corrupt: `When Sandra and Betty went to the store, Arthur gave a drink to`
- controlled clean: `Thus Sandra beside Betty painted every silent garden; Betty carried some token toward`
- controlled corrupt: `Thus Sandra beside Betty painted every silent garden; Arthur carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 078 — exploratory-038-baba

- split/template/order: `exploratory` / `explore-store` / `BABA`
- roles: IO=`Sandra`, subject=`Betty`, corruption=`Arthur`
- canonical clean: `When Betty and Sandra went to the store, Betty gave a drink to`
- canonical corrupt: `When Betty and Sandra went to the store, Arthur gave a drink to`
- controlled clean: `Thus Betty beside Sandra painted every silent garden; Betty carried some token toward`
- controlled corrupt: `Thus Betty beside Sandra painted every silent garden; Arthur carried some token toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 079 — exploratory-039-abba

- split/template/order: `exploratory` / `explore-park` / `ABBA`
- roles: IO=`Linda`, subject=`Sarah`, corruption=`Gary`
- canonical clean: `Then Linda and Sarah went to the park, Sarah handed a book to`
- canonical corrupt: `Then Linda and Sarah went to the park, Gary handed a book to`
- controlled clean: `Today Linda near Sarah crossed each quiet field; Sarah moved one object toward`
- controlled corrupt: `Today Linda near Sarah crossed each quiet field; Gary moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 080 — exploratory-039-baba

- split/template/order: `exploratory` / `explore-park` / `BABA`
- roles: IO=`Linda`, subject=`Sarah`, corruption=`Gary`
- canonical clean: `Then Sarah and Linda went to the park, Sarah handed a book to`
- canonical corrupt: `Then Sarah and Linda went to the park, Gary handed a book to`
- controlled clean: `Today Sarah near Linda crossed each quiet field; Sarah moved one object toward`
- controlled corrupt: `Today Sarah near Linda crossed each quiet field; Gary moved one object toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 081 — development-000-abba

- split/template/order: `development` / `development-office` / `ABBA`
- roles: IO=`Michael`, subject=`Amy`, corruption=`Paul`
- canonical clean: `After Michael and Amy went to the office, Amy sent a letter to`
- canonical corrupt: `After Michael and Amy went to the office, Paul sent a letter to`
- controlled clean: `Later Michael around Amy shaped this green circle; Amy pushed that signal toward`
- controlled corrupt: `Later Michael around Amy shaped this green circle; Paul pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 082 — development-000-baba

- split/template/order: `development` / `development-office` / `BABA`
- roles: IO=`Michael`, subject=`Amy`, corruption=`Paul`
- canonical clean: `After Amy and Michael went to the office, Amy sent a letter to`
- canonical corrupt: `After Amy and Michael went to the office, Paul sent a letter to`
- controlled clean: `Later Amy around Michael shaped this green circle; Amy pushed that signal toward`
- controlled corrupt: `Later Amy around Michael shaped this green circle; Paul pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 083 — development-001-abba

- split/template/order: `development` / `development-office` / `ABBA`
- roles: IO=`Gary`, subject=`George`, corruption=`Sharon`
- canonical clean: `After Gary and George went to the office, George sent a letter to`
- canonical corrupt: `After Gary and George went to the office, Sharon sent a letter to`
- controlled clean: `Later Gary around George shaped this green circle; George pushed that signal toward`
- controlled corrupt: `Later Gary around George shaped this green circle; Sharon pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 084 — development-001-baba

- split/template/order: `development` / `development-office` / `BABA`
- roles: IO=`Gary`, subject=`George`, corruption=`Sharon`
- canonical clean: `After George and Gary went to the office, George sent a letter to`
- canonical corrupt: `After George and Gary went to the office, Sharon sent a letter to`
- controlled clean: `Later George around Gary shaped this green circle; George pushed that signal toward`
- controlled corrupt: `Later George around Gary shaped this green circle; Sharon pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 085 — development-002-abba

- split/template/order: `development` / `development-office` / `ABBA`
- roles: IO=`Anna`, subject=`David`, corruption=`Patricia`
- canonical clean: `After Anna and David went to the office, David sent a letter to`
- canonical corrupt: `After Anna and David went to the office, Patricia sent a letter to`
- controlled clean: `Later Anna around David shaped this green circle; David pushed that signal toward`
- controlled corrupt: `Later Anna around David shaped this green circle; Patricia pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 086 — development-002-baba

- split/template/order: `development` / `development-office` / `BABA`
- roles: IO=`Anna`, subject=`David`, corruption=`Patricia`
- canonical clean: `After David and Anna went to the office, David sent a letter to`
- canonical corrupt: `After David and Anna went to the office, Patricia sent a letter to`
- controlled clean: `Later David around Anna shaped this green circle; David pushed that signal toward`
- controlled corrupt: `Later David around Anna shaped this green circle; Patricia pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 087 — development-003-abba

- split/template/order: `development` / `development-office` / `ABBA`
- roles: IO=`Paul`, subject=`Linda`, corruption=`Gary`
- canonical clean: `After Paul and Linda went to the office, Linda sent a letter to`
- canonical corrupt: `After Paul and Linda went to the office, Gary sent a letter to`
- controlled clean: `Later Paul around Linda shaped this green circle; Linda pushed that signal toward`
- controlled corrupt: `Later Paul around Linda shaped this green circle; Gary pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 088 — development-003-baba

- split/template/order: `development` / `development-office` / `BABA`
- roles: IO=`Paul`, subject=`Linda`, corruption=`Gary`
- canonical clean: `After Linda and Paul went to the office, Linda sent a letter to`
- canonical corrupt: `After Linda and Paul went to the office, Gary sent a letter to`
- controlled clean: `Later Linda around Paul shaped this green circle; Linda pushed that signal toward`
- controlled corrupt: `Later Linda around Paul shaped this green circle; Gary pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 089 — development-004-abba

- split/template/order: `development` / `development-office` / `ABBA`
- roles: IO=`Elizabeth`, subject=`Donna`, corruption=`Barbara`
- canonical clean: `After Elizabeth and Donna went to the office, Donna sent a letter to`
- canonical corrupt: `After Elizabeth and Donna went to the office, Barbara sent a letter to`
- controlled clean: `Later Elizabeth around Donna shaped this green circle; Donna pushed that signal toward`
- controlled corrupt: `Later Elizabeth around Donna shaped this green circle; Barbara pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 090 — development-004-baba

- split/template/order: `development` / `development-office` / `BABA`
- roles: IO=`Elizabeth`, subject=`Donna`, corruption=`Barbara`
- canonical clean: `After Donna and Elizabeth went to the office, Donna sent a letter to`
- canonical corrupt: `After Donna and Elizabeth went to the office, Barbara sent a letter to`
- controlled clean: `Later Donna around Elizabeth shaped this green circle; Donna pushed that signal toward`
- controlled corrupt: `Later Donna around Elizabeth shaped this green circle; Barbara pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 091 — development-005-abba

- split/template/order: `development` / `development-office` / `ABBA`
- roles: IO=`Helen`, subject=`Linda`, corruption=`George`
- canonical clean: `After Helen and Linda went to the office, Linda sent a letter to`
- canonical corrupt: `After Helen and Linda went to the office, George sent a letter to`
- controlled clean: `Later Helen around Linda shaped this green circle; Linda pushed that signal toward`
- controlled corrupt: `Later Helen around Linda shaped this green circle; George pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 092 — development-005-baba

- split/template/order: `development` / `development-office` / `BABA`
- roles: IO=`Helen`, subject=`Linda`, corruption=`George`
- canonical clean: `After Linda and Helen went to the office, Linda sent a letter to`
- canonical corrupt: `After Linda and Helen went to the office, George sent a letter to`
- controlled clean: `Later Linda around Helen shaped this green circle; Linda pushed that signal toward`
- controlled corrupt: `Later Linda around Helen shaped this green circle; George pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 093 — development-006-abba

- split/template/order: `development` / `development-office` / `ABBA`
- roles: IO=`Robert`, subject=`Margaret`, corruption=`Frank`
- canonical clean: `After Robert and Margaret went to the office, Margaret sent a letter to`
- canonical corrupt: `After Robert and Margaret went to the office, Frank sent a letter to`
- controlled clean: `Later Robert around Margaret shaped this green circle; Margaret pushed that signal toward`
- controlled corrupt: `Later Robert around Margaret shaped this green circle; Frank pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 094 — development-006-baba

- split/template/order: `development` / `development-office` / `BABA`
- roles: IO=`Robert`, subject=`Margaret`, corruption=`Frank`
- canonical clean: `After Margaret and Robert went to the office, Margaret sent a letter to`
- canonical corrupt: `After Margaret and Robert went to the office, Frank sent a letter to`
- controlled clean: `Later Margaret around Robert shaped this green circle; Margaret pushed that signal toward`
- controlled corrupt: `Later Margaret around Robert shaped this green circle; Frank pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 095 — development-007-abba

- split/template/order: `development` / `development-office` / `ABBA`
- roles: IO=`Karen`, subject=`George`, corruption=`Lisa`
- canonical clean: `After Karen and George went to the office, George sent a letter to`
- canonical corrupt: `After Karen and George went to the office, Lisa sent a letter to`
- controlled clean: `Later Karen around George shaped this green circle; George pushed that signal toward`
- controlled corrupt: `Later Karen around George shaped this green circle; Lisa pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 096 — development-007-baba

- split/template/order: `development` / `development-office` / `BABA`
- roles: IO=`Karen`, subject=`George`, corruption=`Lisa`
- canonical clean: `After George and Karen went to the office, George sent a letter to`
- canonical corrupt: `After George and Karen went to the office, Lisa sent a letter to`
- controlled clean: `Later George around Karen shaped this green circle; George pushed that signal toward`
- controlled corrupt: `Later George around Karen shaped this green circle; Lisa pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 097 — development-008-abba

- split/template/order: `development` / `development-office` / `ABBA`
- roles: IO=`Jennifer`, subject=`Alice`, corruption=`Sarah`
- canonical clean: `After Jennifer and Alice went to the office, Alice sent a letter to`
- canonical corrupt: `After Jennifer and Alice went to the office, Sarah sent a letter to`
- controlled clean: `Later Jennifer around Alice shaped this green circle; Alice pushed that signal toward`
- controlled corrupt: `Later Jennifer around Alice shaped this green circle; Sarah pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 098 — development-008-baba

- split/template/order: `development` / `development-office` / `BABA`
- roles: IO=`Jennifer`, subject=`Alice`, corruption=`Sarah`
- canonical clean: `After Alice and Jennifer went to the office, Alice sent a letter to`
- canonical corrupt: `After Alice and Jennifer went to the office, Sarah sent a letter to`
- controlled clean: `Later Alice around Jennifer shaped this green circle; Alice pushed that signal toward`
- controlled corrupt: `Later Alice around Jennifer shaped this green circle; Sarah pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 099 — development-009-abba

- split/template/order: `development` / `development-office` / `ABBA`
- roles: IO=`Margaret`, subject=`Elizabeth`, corruption=`Ruth`
- canonical clean: `After Margaret and Elizabeth went to the office, Elizabeth sent a letter to`
- canonical corrupt: `After Margaret and Elizabeth went to the office, Ruth sent a letter to`
- controlled clean: `Later Margaret around Elizabeth shaped this green circle; Elizabeth pushed that signal toward`
- controlled corrupt: `Later Margaret around Elizabeth shaped this green circle; Ruth pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed

## 100 — development-009-baba

- split/template/order: `development` / `development-office` / `BABA`
- roles: IO=`Margaret`, subject=`Elizabeth`, corruption=`Ruth`
- canonical clean: `After Elizabeth and Margaret went to the office, Elizabeth sent a letter to`
- canonical corrupt: `After Elizabeth and Margaret went to the office, Ruth sent a letter to`
- controlled clean: `Later Elizabeth around Margaret shaped this green circle; Elizabeth pushed that signal toward`
- controlled corrupt: `Later Elizabeth around Margaret shaped this green circle; Ruth pushed that signal toward`
- audit status: token and role invariants passed; independent human review not yet signed
