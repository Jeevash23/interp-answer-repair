"""Guard the relocation boundary without changing any retained evidence."""
import hashlib
from activation_transfer import paths


def test_recorded_model_path_resolves_to_shared_checkpoint():
    old=paths.ROOT/'studies/routing/../transport-project/models/gpt2-small/config.json'
    assert paths.recorded_path(old).resolve()==(paths.ROOT/'models/gpt2-small/config.json').resolve()


def test_relocation_never_accepts_an_unrecorded_source_edit(tmp_path,monkeypatch):
    path=tmp_path/'source.py';path.write_text('relocated code\n')
    original=hashlib.sha256(b'original code\n').hexdigest()
    relocated=hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(paths,'ROOT',tmp_path)
    monkeypatch.setattr(paths,'relocation_records',lambda:{'source.py':{'original_sha256':original,'relocated_sha256':relocated}})
    assert paths.frozen_file_matches(path,original)
    path.write_text('unrecorded change\n')
    assert not paths.frozen_file_matches(path,original)
    assert not paths.frozen_file_matches(path,'unrecognized original hash')


def test_modified_measurement_cannot_be_accepted_as_a_relocation(tmp_path,monkeypatch):
    path=tmp_path/'measurement.json';path.write_text('{"value":2}\n')
    monkeypatch.setattr(paths,'ROOT',tmp_path)
    monkeypatch.setattr(paths,'relocation_records',lambda:{})
    assert not paths.frozen_file_matches(path,hashlib.sha256(b'{"value":1}\n').hexdigest())
