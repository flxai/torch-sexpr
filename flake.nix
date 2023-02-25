{
  description = "Parse S-expressions into PyTorch architectures, losses and optimizers";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/695b3515251873e0a7e2021add4bba643c56cde3";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = inputs: with inputs;
  flake-utils.lib.eachDefaultSystem (system: let
    pkgs = import nixpkgs {inherit system;};
  in {
    defaultPackage = self.packages.${system}.torch-sexpr;
    packages.torch-sexpr = pkgs.poetry2nix.mkPoetryApplication {
      projectDir = ./.;
    };
  });
}
